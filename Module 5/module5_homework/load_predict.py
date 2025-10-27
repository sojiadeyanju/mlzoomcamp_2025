import pickle

# Load model
with open('pipeline_v1.bin', 'rb') as file:
    pipeline = pickle.load(file)

# Input as list of dict
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict
proba = pipeline.predict_proba([record])[0][1]

print(f"Conversion Probability: {proba:.4f} ({proba*100:.2f}%)")