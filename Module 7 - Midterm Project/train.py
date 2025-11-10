import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURATION AND DATA LOADING ---
DATA_PATH = 'crop_yield.csv'
MODEL_FILENAME = 'xgb_yield_model.joblib'
FEATURES_FILENAME = 'model_features.joblib'
TARGET_COLUMN = 'Yield_tons_per_hectare'
RANDOM_STATE = 42

print(f"Starting training process using {DATA_PATH}...")

try:
    # Load the data safely (using a sample to prevent memory issues)
    df = pd.read_csv(DATA_PATH, nrows=100000, 
                    dtype={
                        'Rainfall_mm': np.float32, 
                        'Temperature_Celsius': np.float32,
                        'Days_to_Harvest': np.int16,
                        'Yield_tons_per_hectare': np.float32
                    })

    # Data Type Cleanup and Missing Value Handling
    df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)
    df['Irrigation_Used'] = df['Irrigation_Used'].astype(str).str.lower().map({'true': 1, 'false': 0}).fillna(0).astype(int)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    
except Exception as e:
    print(f"Error loading or cleaning data: {e}")
    exit()

# --- 2. PREPROCESSING (ONE-HOT ENCODING) ---
nominal_features = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
df_encoded = pd.get_dummies(df, columns=nominal_features, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop(TARGET_COLUMN, axis=1)
y = df_encoded[TARGET_COLUMN]

# Save the exact list of feature names for the prediction service
feature_names = X.columns.tolist()
joblib.dump(feature_names, FEATURES_FILENAME)
print(f"Feature list saved to {FEATURES_FILENAME}")

# Split data (just for evaluation, model is trained on full train set later)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# --- 3. MODEL TRAINING AND SAVING ---
print("Training final XGBoost Regressor...")

# Use the best model configuration found in the previous step
final_model = XGBRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=10, 
    random_state=RANDOM_STATE, 
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# Evaluate the final model
y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nFinal Model Performance (Test Set):")
print(f"  R2 Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Save the trained model
joblib.dump(final_model, MODEL_FILENAME)
print(f"Trained model saved successfully to {MODEL_FILENAME}")