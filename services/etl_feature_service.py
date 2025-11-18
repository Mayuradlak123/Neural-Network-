import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from config.logger import logger
from datetime import datetime
import json


class ETLFeatureService:
    """
    Service for ETL operations and Feature Engineering on CSV files
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.transformations_history: List[Dict] = []
    
    def load_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Load CSV file and extract basic information
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dict with data info and preview
        """
        try:
            logger.info(f"Loading CSV from: {file_path}")
            
            # Load CSV
            self.df = pd.read_csv(file_path)
            self.original_df = self.df.copy()
            self.file_path = file_path
            
            # Get basic info
            info = self._get_data_info()
            
            logger.info(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
            return {
                "success": True,
                "file_path": file_path,
                "info": info
            }
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    def _serialize_documents(self, documents: List[Dict]) -> List[Dict]:
        """Convert documents to JSON-serializable format"""
        serialized = []
        for doc in documents:
            serialized_doc = {}
            for key, value in doc.items():
                # Handle NaN values
                if pd.isna(value):
                    serialized_doc[key] = None
                elif hasattr(value, '__dict__'):  # ObjectId, datetime, etc.
                    serialized_doc[key] = str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    # Convert numpy types to Python native types
                    if np.isnan(value):
                        serialized_doc[key] = None
                    else:
                        serialized_doc[key] = value.item()
                else:
                    serialized_doc[key] = value
            serialized.append(serialized_doc)
        return serialized
    
    def _get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive data information"""
        if self.df is None:
            return {}
        
        # Basic stats
        info = {
            "rows": int(len(self.df)),
            "columns": int(len(self.df.columns)),
            "column_names": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {k: int(v) for k, v in self.df.isnull().sum().to_dict().items()},
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "preview": self._serialize_documents(self.df.head(10).to_dict(orient='records'))
        }
        
        # Numeric columns stats
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = self.df[numeric_cols].describe()
            # Replace NaN with None for JSON compatibility
            info["numeric_stats"] = json.loads(stats_df.to_json())
        
        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            info["categorical_info"] = {
                col: {
                    "unique_values": int(self.df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in self.df[col].value_counts().head(5).to_dict().items()}
                }
                for col in categorical_cols
            }
        
        return info
    
    def handle_missing_values(self, strategy: str, columns: List[str] = None) -> Dict[str, Any]:
        """
        Handle missing values using different strategies
        
        Args:
            strategy: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'constant'
            columns: List of columns to apply (None = all)
            
        Returns:
            Dict with results
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Handling missing values with strategy: {strategy}")
            
            cols = columns if columns else list(self.df.columns)
            before_missing = self.df[cols].isnull().sum().sum()
            
            if strategy == 'drop':
                self.df = self.df.dropna(subset=cols)
            elif strategy == 'mean':
                for col in cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                for col in cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                for col in cols:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == 'forward_fill':
                self.df[cols] = self.df[cols].fillna(method='ffill')
            elif strategy == 'backward_fill':
                self.df[cols] = self.df[cols].fillna(method='bfill')
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            after_missing = self.df[cols].isnull().sum().sum()
            
            self.transformations_history.append({
                "operation": "handle_missing_values",
                "strategy": strategy,
                "columns": cols,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "before_missing": int(before_missing),
                "after_missing": int(after_missing),
                "rows_remaining": len(self.df)
            }
            
        except Exception as e:
            logger.error(f"Failed to handle missing values: {e}")
            raise
    
    def encode_categorical(self, columns: List[str], method: str = 'label') -> Dict[str, Any]:
        """
        Encode categorical variables
        
        Args:
            columns: List of columns to encode
            method: 'label' or 'onehot'
            
        Returns:
            Dict with encoding info
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Encoding categorical columns: {columns} with method: {method}")
            
            if method == 'label':
                from sklearn.preprocessing import LabelEncoder
                encoders = {}
                for col in columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    encoders[col] = list(le.classes_)
                
                result = {"encoders": encoders}
                
            elif method == 'onehot':
                self.df = pd.get_dummies(self.df, columns=columns, prefix=columns)
                result = {"new_columns": [col for col in self.df.columns if any(prefix in col for prefix in columns)]}
            else:
                raise ValueError(f"Unknown encoding method: {method}")
            
            self.transformations_history.append({
                "operation": "encode_categorical",
                "method": method,
                "columns": columns,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "method": method,
                **result,
                "new_shape": self.df.shape
            }
            
        except Exception as e:
            logger.error(f"Failed to encode categorical: {e}")
            raise
    
    def scale_features(self, columns: List[str], method: str = 'standard') -> Dict[str, Any]:
        """
        Scale numeric features
        
        Args:
            columns: List of columns to scale
            method: 'standard', 'minmax', 'robust'
            
        Returns:
            Dict with scaling info
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Scaling features: {columns} with method: {method}")
            
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            self.df[columns] = scaler.fit_transform(self.df[columns])
            
            self.transformations_history.append({
                "operation": "scale_features",
                "method": method,
                "columns": columns,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "method": method,
                "columns": columns,
                "stats": self.df[columns].describe().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            raise
    
    def create_features(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create new features based on operations
        
        Args:
            operations: List of feature creation operations
                Example: [
                    {"type": "polynomial", "column": "age", "degree": 2},
                    {"type": "interaction", "columns": ["age", "income"]},
                    {"type": "binning", "column": "age", "bins": 5}
                ]
                
        Returns:
            Dict with created features
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Creating {len(operations)} new features")
            
            created_features = []
            
            for op in operations:
                op_type = op.get("type")
                
                if op_type == "polynomial":
                    col = op["column"]
                    degree = op.get("degree", 2)
                    new_col = f"{col}_pow{degree}"
                    self.df[new_col] = self.df[col] ** degree
                    created_features.append(new_col)
                
                elif op_type == "interaction":
                    cols = op["columns"]
                    new_col = f"{'_x_'.join(cols)}"
                    self.df[new_col] = self.df[cols[0]]
                    for col in cols[1:]:
                        self.df[new_col] *= self.df[col]
                    created_features.append(new_col)
                
                elif op_type == "binning":
                    col = op["column"]
                    bins = op.get("bins", 5)
                    new_col = f"{col}_binned"
                    self.df[new_col] = pd.cut(self.df[col], bins=bins, labels=False)
                    created_features.append(new_col)
                
                elif op_type == "log":
                    col = op["column"]
                    new_col = f"{col}_log"
                    self.df[new_col] = np.log1p(self.df[col])
                    created_features.append(new_col)
                
                elif op_type == "sqrt":
                    col = op["column"]
                    new_col = f"{col}_sqrt"
                    self.df[new_col] = np.sqrt(self.df[col].abs())
                    created_features.append(new_col)
            
            self.transformations_history.append({
                "operation": "create_features",
                "operations": operations,
                "created_features": created_features,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "created_features": created_features,
                "total_columns": len(self.df.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            raise
    
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Remove outliers from specified columns
        
        Args:
            columns: List of columns to check
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            Dict with outlier removal info
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Removing outliers from: {columns} using method: {method}")
            
            before_rows = len(self.df)
            
            if method == 'iqr':
                for col in columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            
            elif method == 'zscore':
                from scipy import stats
                for col in columns:
                    z_scores = np.abs(stats.zscore(self.df[col]))
                    self.df = self.df[z_scores < threshold]
            
            after_rows = len(self.df)
            
            self.transformations_history.append({
                "operation": "remove_outliers",
                "method": method,
                "columns": columns,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "before_rows": before_rows,
                "after_rows": after_rows,
                "removed_rows": before_rows - after_rows
            }
            
        except Exception as e:
            logger.error(f"Failed to remove outliers: {e}")
            raise
    
    def export_data(self, output_path: str) -> Dict[str, Any]:
        """
        Export processed data to CSV
        
        Args:
            output_path: Path to save the file
            
        Returns:
            Dict with export info
        """
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            logger.info(f"Exporting data to: {output_path}")
            
            self.df.to_csv(output_path, index=False)
            
            return {
                "success": True,
                "output_path": output_path,
                "rows": len(self.df),
                "columns": len(self.df.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current dataframe state"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        return self._get_data_info()
    
    def get_transformations_history(self) -> List[Dict]:
        """Get history of all transformations"""
        return self.transformations_history
    
    def reset_to_original(self) -> Dict[str, Any]:
        """Reset to original loaded data"""
        if self.original_df is None:
            raise ValueError("No original data available")
        
        self.df = self.original_df.copy()
        self.transformations_history = []
        
        return {
            "success": True,
            "message": "Data reset to original state",
            "info": self._get_data_info()
        }


# Session storage
_etl_sessions = {}


def get_etl_session(session_id: str) -> ETLFeatureService:
    """Get or create an ETL session"""
    if session_id not in _etl_sessions:
        _etl_sessions[session_id] = ETLFeatureService()
    return _etl_sessions[session_id]


def cleanup_etl_session(session_id: str):
    """Clean up an ETL session"""
    if session_id in _etl_sessions:
        del _etl_sessions[session_id]