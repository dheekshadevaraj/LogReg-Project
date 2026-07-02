# ==========================================
# Diabetes Prediction using Logistic Regression
# ==========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Step 2: Load Dataset
# ==========================================

df = pd.read_csv(r"C:\Users\admin\Downloads\archive (1)\diabetes.csv")

# ==========================================
# Step 3: Basic Exploration
# ==========================================

print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# ==========================================
# Step 4: Features and Target
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==========================================
# Step 5: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Step 6: Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Step 7: Create Logistic Regression Model
# ==========================================

model = LogisticRegression(random_state=42)

# ==========================================
# Step 8: Train the Model
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# Step 9: Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 10: Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ==========================================
# Step 11: Confusion Matrix
# ==========================================

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# Step 12: Classification Report
# ==========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 13: Predict New Patient
# ==========================================

sample = [[2,120,70,30,85,32.5,0.45,28]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("\nPrediction: Diabetic")
else:
    print("\nPrediction: Not Diabetic")

# ==========================================
# Step 14: Prediction Probability
# ==========================================

probability = model.predict_proba(sample_scaled)

print("\nProbability of Not Diabetic:", round(probability[0][0] * 100, 2), "%")
print("Probability of Diabetic:", round(probability[0][1] * 100, 2), "%")

# ==========================================
# Step 15: Save Model and Scaler
# ==========================================

import pickle

pickle.dump(model, open("logistic_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel and Scaler saved successfully!")