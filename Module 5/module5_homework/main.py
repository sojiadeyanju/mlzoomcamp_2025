# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Lead Conversion Prediction API",
    description="Predicts probability of lead converting to a paid subscription",
    version="1.0"
)

# Load model at startup (once)
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        with open("pipeline_v1.bin", "rb") as f:
            pipeline = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Define input schema
class LeadInput(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Prediction response
class PredictionOutput(BaseModel):
    conversion_probability: float
    message: str = "Prediction successful"

# Root endpoint
@app.get("/")
def home():
    return {"message": "Lead Conversion API is running. Use POST /predict"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(lead: LeadInput):
    try:
        # Convert to list of dicts (required for DictVectorizer)
        input_data = [{
            "lead_source": lead.lead_source,
            "number_of_courses_viewed": lead.number_of_courses_viewed,
            "annual_income": lead.annual_income
        }]

        # Get probability of class 1 (conversion)
        prob = pipeline.predict_proba(input_data)[0][1]

        return PredictionOutput(
            conversion_probability=round(float(prob), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))