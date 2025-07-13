import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# Load CSV
csv_file_path = 'F:\\ML project\\data.csv.csv'
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"CSV file not found at path: {csv_file_path}")

df = pd.read_csv(csv_file_path)

# Rename duplicate columns if needed
original_cols = df.columns.tolist()
if len(set(original_cols)) != len(original_cols):
    new_cols = [f"prev_sem_cgpa" if i < len(original_cols) - 1 else 'next_sem_cgpa' for i in range(len(original_cols))]
    df.columns = new_cols

print("Renamed Columns:", df.columns.tolist())

# Drop missing values
df.dropna(inplace=True)
print("Cleaned Row Count:", len(df))
if df.empty:
    raise ValueError("DataFrame is empty after removing missing values.")

# Feature and target separation
X = df.drop('next_sem_cgpa', axis=1)
y = df['next_sem_cgpa']

# Ensure all features are numeric
if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
    raise ValueError("Non-numeric features found in the input data.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf', C=10, epsilon=0.1)
}

best_model = None
best_score = -np.inf

print("\nModel Evaluation Results:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}: R² = {score:.4f}, MSE = {mse:.4f}")
    if score > best_score:
        best_model = model
        best_score = score
        best_model_name = name
        best_y_pred = y_pred

# Save best model and scaler
model_path = "cgpa_best_model.pkl"
scaler_path = "cgpa_scaler.pkl"
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nBest Model: {best_model_name} with R² Score = {best_score:.4f}")
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")

# Show coefficients or feature importances
if hasattr(best_model, 'coef_'):
    print("\nFeature Coefficients:")
    for feature, coef in zip(X.columns, best_model.coef_):
        print(f"{feature}: {coef:.4f}")
elif hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importances:")
    for feature, importance in zip(X.columns, best_model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

# Show sample predictions
print("\nSample Predictions (Actual vs Predicted):")
for actual, pred in zip(y_test[:5], best_y_pred[:5]):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")
