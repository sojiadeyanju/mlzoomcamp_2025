import pandas as pd
import joblib
from flask import Flask, request, jsonify
from io import StringIO
import numpy as np

app = Flask(__name__)

# --- Configuration and Model Loading ---
MODEL_PATH = 'xgb_yield_model.joblib'
FEATURES_PATH = 'model_features.joblib'

try:
    # Load the trained model and the feature list used during training
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    # Exit or handle error if model files are missing
    model = None
    model_features = []

# Features that need one-hot encoding
NOMINAL_FEATURES = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
BOOLEAN_FEATURES = ['Fertilizer_Used', 'Irrigation_Used']
NUMERICAL_FEATURES = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input with raw feature values and returns a yield prediction.
    Example expected input:
    {
        "Region": "North",
        "Soil_Type": "Loam",
        "Crop": "Maize",
        "Rainfall_mm": 550,
        "Temperature_Celsius": 28,
        "Fertilizer_Used": true,
        "Irrigation_Used": true,
        "Days_to_Harvest": 120,
        "Weather_Condition": "Sunny"
    }
    """
    if not model:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # Get raw data from the request
        data = request.get_json(force=True)
        
        # Convert the single input dictionary into a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # 1. Handle Boolean/Binary features (convert True/False to 1/0)
        for col in BOOLEAN_FEATURES:
            input_df[col] = input_df[col].astype(int)

        # 2. Perform One-Hot Encoding consistent with training data
        input_df_encoded = pd.get_dummies(input_df, columns=NOMINAL_FEATURES)
        
        # 3. Align columns: Add missing dummy columns and drop extra ones
        # This step ensures the feature vector matches the model's training structure
        final_input = pd.DataFrame(0, index=[0], columns=model_features)

        # Copy over values that exist in the input and the model's expected features
        for col in final_input.columns:
            if col in input_df_encoded.columns:
                final_input[col] = input_df_encoded[col].iloc[0]
            # Handle numerical features that were not one-hot encoded
            elif col in NUMERICAL_FEATURES or col in BOOLEAN_FEATURES:
                final_input[col] = input_df[col].iloc[0]

        # Ensure order is correct before predicting
        final_input = final_input[model_features]

        # 4. Predict the yield
        prediction = model.predict(final_input)
        
        # Return the prediction result
        return jsonify({
            'status': 'success',
            'predicted_yield_tons_per_hectare': round(prediction[0], 2),
            'model': 'XGBoost Regressor'
        })

    except Exception as e:
        # Log the error and return a detailed message
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "An error occurred during prediction processing.", "details": str(e)}), 400

if __name__ == '__main__':
    # Set host to 0.0.0.0 for Docker compatibility
    app.run(host='0.0.0.0', port=8080, debug=True)