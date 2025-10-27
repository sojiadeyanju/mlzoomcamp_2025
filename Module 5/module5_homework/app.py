# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

app = FastAPI(
    title="Lead Conversion API",
    version="1.0"
)

# Load model once at startup
@app.on_event("startup")
def load_model():
    global pipeline
    with open("pipeline_v1.bin", "rb") as f:
        pipeline = pickle.load(f)

# Input schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Response schema
class Pred(BaseModel):
    conversion_probability: float

@app.post("/predict", response_model=Pred)
def predict(lead: Lead):
    data = [{k: v for k, v in lead.dict().items()}]
    prob = pipeline.predict_proba(data)[0][1]
    return Pred(conversion_probability=round(float(prob), 4))