from typing import Optional,List,Dict,Any
from pydantic import BaseModel


class PredictRequest(BaseModel):
    model_path: str

    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int

    furnishingstatus_semi_furnished: Optional[int] = 0
    furnishingstatus_unfurnished: Optional[int] = 0

class LoadCSVRequest(BaseModel):
    file_path: str


class MissingValuesRequest(BaseModel):
    session_id: str
    strategy: str  # drop, mean, median, mode, forward_fill, backward_fill
    columns: Optional[List[str]] = None


class EncodeCategoricalRequest(BaseModel):
    session_id: str
    columns: List[str]
    method: str = 'label'  # label or onehot


class ScaleFeaturesRequest(BaseModel):
    session_id: str
    columns: List[str]
    method: str = 'standard'  # standard, minmax, robust


class CreateFeaturesRequest(BaseModel):
    session_id: str
    operations: List[Dict[str, Any]]


class RemoveOutliersRequest(BaseModel):
    session_id: str
    columns: List[str]
    method: str = 'iqr'  # iqr or zscore
    threshold: float = 1.5


class ExportDataRequest(BaseModel):
    session_id: str
    output_path: str

class EvaluateRequest(BaseModel):
    model_path: str
    test_data_path: str