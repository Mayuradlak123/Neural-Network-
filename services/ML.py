import pandas as pd

def prepare_data(csv_path, output_path="processed_data.csv"):
    df = pd.read_csv(csv_path)

    # Target variable
    y = df["price"]

    # Features
    X = df.drop("price", axis=1)

    # Convert yes/no to 1/0
    yes_no_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                   "airconditioning", "prefarea"]
    for col in yes_no_cols:
        X[col] = X[col].map({"yes": 1, "no": 0})

    # One-hot encode categorical
    X = pd.get_dummies(X, columns=["furnishingstatus"], drop_first=True)

    # Combine processed X + y to save
    processed_df = X.copy()
    processed_df["price"] = y

    # Save processed CSV
    processed_df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "message": "Data processed and saved successfully",
        "processed_file": output_path,
        "rows": len(processed_df),
        "columns": list(processed_df.columns),
        "sample": processed_df.head(3).to_dict(orient="records")  # small preview in JSON
    }
