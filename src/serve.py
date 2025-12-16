# src/serve.py
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Secure Fraud Detection API")

# Global variable to hold the model
model = None

class Transaction(BaseModel):
    # Simplified: in reality, include all V1-V28 columns
    features: list 

@app.on_event("startup")
def load_model():
    global model
    # Replace with your local MLflow run ID or registered model name
    model_uri = "models:/FraudDetectionModel/Production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except:
        print("Model not found in registry. Serving dummy predictions.")

@app.post("/predict")
def predict(data: Transaction):
    if model is None:
        return {"prediction": 0, "status": "No model loaded"}
    
    # Convert input to DataFrame for the model
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)
    return {"fraud_prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)