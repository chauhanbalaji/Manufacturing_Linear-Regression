import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv("manufacturing_dataset_1000_samples.csv")

# Drop timestamp if present
df = df.iloc[:, 1:]
# df.drop(columns=['timestamp']) Safer alternative (if you know the column name.)

# --- Select only numeric columns for features ---
numeric_cols = ["Injection_Temperature", "Injection_Pressure", "Material_Viscosity"]
X = df[numeric_cols].copy()
y = df["Parts_Per_Hour"]

# Handle missing values (numeric only)
X = X.fillna(X.mean(numeric_only=True))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Predictions
y_pred = model.predict(X_test)

# --- Evaluation Metrics ---
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Metrics:")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

print("Model saved as linear_regression_model.pkl")


