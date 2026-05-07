#  DiagnoWeb — FastAPI Backend

from fastapi import FastAPI, HTTPException

# CORSMiddleware = allows our HTML frontend (different port) to talk to this API
from fastapi.middleware.cors import CORSMiddleware

# BaseModel = lets us define what shape our request/response data should be
from pydantic import BaseModel, Field

# joblib = loads our saved model and scaler from .pkl files
import joblib

import pandas as pd
# os = helps us build file paths that work on any operating system
import os

# datetime = to add timestamp to each prediction (for history)
from datetime import datetime

# List = type hint for Python lists (used in response)
from typing import List

#  Create the FastAPI app
app = FastAPI()

#  Setup CORS
# CORS = Cross-Origin Resource Sharing
# Without this, your browser blocks the frontend from calling the API
# allow_origins=["*"] means "accept requests from ANY website"
# In a real production app, replace * with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],   # allow any headers
)


#  Load the trained model and scaler
# __file__  = path of this current file (main.py)
# dirname() = gets the folder that contains main.py
# join()    = combines folder path + filename

# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
model_path="model.pkl"
scaler_path="scaler.pkl"
# Try to load the files. If they don't exist, show a clear error.
try:
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully!")
except FileNotFoundError:
    # This error will show in terminal if you forgot to train the model first
    raise RuntimeError("❌ model.pkl or scaler.pkl not found! Run model/train.py first.")



#  Pydantic Models (defines what data looks like)

# This defines what data the user SENDS to us (request body)
# Each field has: type, example value, description, and min value (ge=0)
class PatientInput(BaseModel):
    Pregnancies:              float = Field(..., example=2,     ge=0, description="Number of pregnancies")
    Glucose:                  float = Field(..., example=120,   ge=0, description="Plasma glucose mg/dL")
    BloodPressure:            float = Field(..., example=70,    ge=0, description="Diastolic blood pressure mm Hg")
    SkinThickness:            float = Field(..., example=20,    ge=0, description="Triceps skin fold thickness mm")
    Insulin:                  float = Field(..., example=80,    ge=0, description="2-Hour serum insulin mu U/ml")
    BMI:                      float = Field(..., example=25.5,  ge=0, description="Body mass index kg/m²")
    DiabetesPedigreeFunction: float = Field(..., example=0.5,   ge=0, description="Diabetes heredity score")
    Age:                      float = Field(..., example=30,    ge=0, description="Age in years")

# This defines what data we SEND BACK to the user (response body)
class PredictionResponse(BaseModel):
    prediction:   int         # 0 = Healthy, 1 = Diabetic
    result:       str         # "Diabetic" or "Not Diabetic"
    confidence:   float       # how confident the model is (0-100%)
    risk_level:   str         # "High Risk", "Moderate Risk", "Low Risk"
    risk_score:   int         # 0-100 score
    insights:     List[str]   # list of health tips based on input
    timestamp:    str         # when this prediction was made



#  Helper function: Generate health insights
# This runs AFTER the model predicts, to give useful feedback to the user
def generate_insights(data: PatientInput, prediction: int) -> List[str]:
    tips = []  # start with empty list, we'll add tips based on conditions

    # Glucose insight
    if data.Glucose > 140:
        tips.append("⚠️ Your glucose level is high. Consider consulting a doctor for a glucose tolerance test.")
    elif data.Glucose < 70:
        tips.append("⚠️ Low glucose detected. Eat regular meals and monitor your blood sugar.")
    else:
        tips.append("✅ Your glucose level is in the normal range. Keep maintaining a balanced diet.")

    # BMI insight
    if data.BMI > 30:
        tips.append("⚠️ BMI indicates obesity. Regular exercise and a calorie-controlled diet can significantly reduce diabetes risk.")
    elif data.BMI > 25:
        tips.append("ℹ️ BMI is slightly high. Light cardio 3-4 times a week can help.")
    else:
        tips.append("✅ BMI is healthy. Maintain your current weight through regular activity.")

    # Age insight
    if data.Age > 45:
        tips.append("ℹ️ Risk increases with age. Annual blood sugar screenings are recommended after 45.")

    # Blood pressure insight
    if data.BloodPressure > 90:
        tips.append("⚠️ High blood pressure detected. Reduce salt intake and manage stress.")

    # Pedigree (family history) insight
    if data.DiabetesPedigreeFunction > 0.8:
        tips.append("ℹ️ Strong family history of diabetes. Preventive lifestyle changes are highly recommended.")

    # Final tip based on prediction
    if prediction == 1:
        tips.append("🔴 High-risk prediction. Please schedule a doctor's appointment soon.")
    else:
        tips.append("🟢 Low risk detected. Keep up your healthy habits!")

    return tips


#  Helper function: Calculate risk score (0-100)
def calculate_risk_score(data: PatientInput, confidence: float, prediction: int) -> int:
    score = 0

    # Each risk factor adds points
    if data.Glucose > 140:     score += 25
    elif data.Glucose > 100:   score += 12

    if data.BMI > 30:          score += 20
    elif data.BMI > 25:        score += 10

    if data.Age > 45:          score += 15
    elif data.Age > 35:        score += 8

    if data.BloodPressure > 90: score += 10

    if data.DiabetesPedigreeFunction > 0.8: score += 10

    if data.Pregnancies > 5:   score += 5

    if data.Insulin > 200:     score += 5

    # Cap score at 100
    return min(score, 100)


#  API Routes (Endpoints)


#home route
@app.get("/")
def root():
    return {
        "message": "DiagnoWeb API is running!",
    }


# GET /health → used by frontend to check if backend is alive
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


# POST /predict → main endpoint — receives patient data, returns prediction
# POST means the user is SENDING data to us (not just reading)
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientInput):
    # FastAPI automatically reads the JSON body and maps it to PatientInput
    # If any field is missing or wrong type, FastAPI auto-rejects with error

    try:
        # Step 1: Convert input to numpy array (what scikit-learn expects)
        # Shape must be (1, 8) — 1 row, 8 features

        features = pd.DataFrame([{
            "Pregnancies": data.Pregnancies,
            "Glucose": data.Glucose,
            "BloodPressure": data.BloodPressure,
            "SkinThickness": data.SkinThickness,
            "Insulin": data.Insulin,
            "BMI": data.BMI,
            "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
            "Age": data.Age
        }])

        # Step 2: Scale the input USING the same scaler from training
        # Very important — must use same scaler, not re-fit!
        features_scaled = scaler.transform(features)

        # Step 3: Make prediction
        # model.predict() returns [0] or [1]
        # [0] at the end gets the first (only) value from the array
        prediction = int(model.predict(features_scaled)[0])

        # Step 4: Get confidence (probability of the predicted class)
        # predict_proba() returns [[prob_class0, prob_class1]]
        # We take the probability of the PREDICTED class
        probability = float(model.predict_proba(features_scaled)[0][prediction])
        confidence  = round(probability * 100, 1)  # convert to percentage

        # Step 5: Determine result label and risk level
        if prediction == 1:
            result = "Diabetic"
            risk_level = "High Risk" if confidence >= 75 else "Moderate Risk"
        else:
            result = "Not Diabetic"
            risk_level = "Low Risk" if confidence >= 75 else "Borderline"

        # Step 6: Generate insights and risk score
        insights   = generate_insights(data, prediction)
        risk_score = calculate_risk_score(data, confidence , prediction)

        # Step 7: Return the full response
        return PredictionResponse(
            prediction=prediction,
            result=result,
            confidence=confidence,
            risk_level=risk_level,
            risk_score=risk_score,
            insights=insights,
            timestamp=datetime.now().strftime("%d %b %Y, %I:%M %p")
        )

    except Exception as e:
        # If anything goes wrong, return a 500 error with the message
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
