#  STEP 1: Import libraries (tools we need)
import pandas as pd          # pandas = work with tables/CSV files
import numpy as np           # numpy  = math operations
import joblib                # joblib = save/load our trained model
import os                    # os     = work with file paths
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split   # split data into train & test
from sklearn.ensemble import RandomForestClassifier    # our ML model
from sklearn.preprocessing import StandardScaler       # scale/normalize the numbers
from sklearn.metrics import accuracy_score, classification_report  # check how good model is

#  STEP 2: Load the dataset
# os.path.dirname(__file__) = folder where THIS file lives
# os.path.join = joins folder path + filename together
# csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
csv_path = "diabetes.csv"# Check if file exists before loading
if not os.path.exists(csv_path):
    print("❌ diabetes.csv not found!")
    print("👉 Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    exit()

# Read the CSV into a DataFrame (think of it as an Excel table in Python)
df = pd.read_csv(csv_path)
print(f"✅ Dataset loaded! Shape: {df.shape}")  # shows rows x columns
print(f"   Diabetic patients : {df['Outcome'].sum()}")
print(f"   Healthy patients  : {len(df) - df['Outcome'].sum()}")

#  STEP 3: Clean the data
# In this dataset, some columns have 0 which is medically impossible
# e.g. Glucose = 0 means no data, not actually 0
# So we replace 0 with NaN (empty), then fill with median (middle value)

cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Replace 0 → NaN in those columns
df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

# Fill missing values with median of each column
df.fillna(df.median(numeric_only=True), inplace=True)

print("✅ Data cleaned (invalid zeros replaced with median values)")

#  STEP 4: Split data into Features (X) and Label (y)
# X = input columns (what we give the model to predict from)
# y = output column (what we want to predict: 0=healthy, 1=diabetic)

X = df.drop("Outcome", axis=1)   # everything EXCEPT Outcome
y = df["Outcome"]                 # only the Outcome column

print(f"✅ Features (X): {list(X.columns)}")
print(f"✅ Label    (y): Outcome (0=Healthy, 1=Diabetic)")

#  STEP 5: Split into Training set and Testing set
# test_size=0.2  → 20% data for testing, 80% for training
# random_state=42 → fixes randomness so results are same every run
# stratify=y → keeps same ratio of 0s and 1s in both splits

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train size: {len(X_train)} rows | Test size: {len(X_test)} rows")

#  STEP 6: Scale the features
# ML models work better when all numbers are on the same scale
# StandardScaler converts everything to mean=0, std=1
# Example: Age 30 → -0.5, Glucose 150 → 1.2 (relative values)

scaler = StandardScaler()

# fit_transform on train = learn the scale FROM train data, then apply it
X_train_scaled = scaler.fit_transform(X_train)

# transform only on test = apply SAME scale (don't learn from test!)
X_test_scaled  = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")

#  STEP 7: Train the ML Model
# RandomForestClassifier = builds many decision trees and combines them
# n_estimators=150 → build 150 trees (more trees = more accurate but slower)
# random_state=42  → fixes randomness

model = RandomForestClassifier(n_estimators=150, random_state=42)

# .fit() = actually TRAIN the model (this is the "learning" step)
model.fit(X_train_scaled, y_train)

print("✅ Model trained!")

#  STEP 8: Evaluate the model
# Use test data to check how good our model is

y_pred = model.predict(X_test_scaled)   # model makes predictions on test data

accuracy = accuracy_score(y_test, y_pred)  # compare predictions vs real answers
print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📊 Full Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Diabetic"]))


# STEP 9: Create performance diagram image

# confusion_matrix = shows correct and wrong predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# matplotlib = used to create charts/images
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create display object for the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Diabetic"]
)

# Plot the confusion matrix chart
display.plot()

# Add chart title with accuracy percentage
plt.title(f"Diabetes Prediction Model\nAccuracy: {accuracy*100:.2f}%")

# Save chart as image file
plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")

# Show chart window
plt.show()

print("✅ model_performance.png saved!")

#  STEP 9: Save the model and scaler to disk
# We save them as .pkl files so FastAPI can load and use them

# save_dir = os.path.dirname(__file__)   # same folder as this script

# joblib.dump(model,  os.path.join(save_dir, "model.pkl"))    # save model
# joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))   # save scaler

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ model.pkl  saved!")
print("✅ scaler.pkl saved!")
print("\n🚀 Now run: uvicorn main:app --reload  (from backend/ folder)")
