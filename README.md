# 🩺 DiagnoWeb — AI-Powered Diabetes Risk Predictor

DiagnoWeb is an end-to-end Machine Learning web application that predicts diabetes risk using patient health metrics.

Built using:
- FastAPI
- Scikit-learn
- Random Forest Classifier
- HTML/CSS/JavaScript
- Chart.js

---

## 🚀 Features

- Real-time diabetes prediction
- AI-generated health insights
- Confidence & risk scoring
- Interactive charts and analytics
- PDF report generation
- FastAPI backend API
- Responsive modern UI

---

## 📊 ML Model

The model is trained on the Pima Indians Diabetes Dataset using a Random Forest Classifier.

### Input Features
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## ⚙️ Tech Stack

### Backend
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

---

## 🧠 API Endpoint

### Predict Diabetes Risk

```http
POST /predict
```

---

## 📦 Installation

```bash
git clone https://github.com/yashbhanushali17/DiagnoWeb.git
cd DiagnoWeb
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 🌐 Deployment

Backend deployed using Render.

---

## 📌 Disclaimer

This project is developed for educational and research purposes only and should not be considered medical advice.

---

## 👨‍💻 Author

Yash Bhanushali
