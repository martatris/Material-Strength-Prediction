# ================================================================
# MACHINE LEARNING: Predict Tensile Strength from Material Features
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib

# ----------------------------
# 1. Load Data
# ----------------------------
file_path = 'X_bp.xlsx'
df = pd.read_excel(file_path)

print("Data Loaded Successfully!\n")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# 2. Define Feature and Target
# ----------------------------
target_col = 'Tensile Strength, Mpa'
X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ----------------------------
# 4. Define Multiple Models
# ----------------------------
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Support Vector Regressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=10, kernel="rbf"))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ]),

    "XGBoost Regressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ))
    ])
}

# ----------------------------
# 5. Train & Evaluate Models
# ----------------------------
results = []

best_model = None
best_r2 = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R² Score"])
print("\nMODEL PERFORMANCE COMPARISON:\n")
print(results_df.sort_values(by="R² Score", ascending=False))

# ----------------------------
# 6. Feature Importance from Random Forest
# ----------------------------
rf = models["Random Forest"].named_steps["model"]
importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Save Best Model
# ----------------------------
joblib.dump(best_model, f"best_model_{best_name.replace(' ', '_')}.pkl")
print(f"\nBest model saved as: best_model_{best_name.replace(' ', '_')}.pkl")
print(f"Best Model: {best_name} (R² = {best_r2:.3f})")
