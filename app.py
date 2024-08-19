from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# Load the model from the MLflow model registry
model = mlflow.sklearn.load_model("mlruns/283525219985850560/77c590f90dd846da808a0fd31b604c6f/artifacts/pipeline_random_forest")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the input data structure
class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# Endpoint to get predictions
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Predict using the loaded model
        prediction = model.predict(input_df)[0]

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
