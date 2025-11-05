# Material Property Prediction Models

This project builds and compares multiple machine learning models to predict the **Tensile Strength (MPa)** of composite materials based on input formulation and material properties.

## Input Data
Your dataset must be stored in an Excel file named:
```
X_bp.xlsx
```
Required columns:
- Matrix-to-filler Ratio
- Density, kg/m^3
- Elastic Modulus, Gpa
- Harderner content, wt.%
- Epoxy group content, %
- Flash point, C
- Surface Density, g/m^3
- Tensile modulus, Gpa
- Tensile Strength, Mpa  ← Target variable
- Resin Consumption, g/m^2

## Model Comparison
The script trains and compares the following models:
- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor
- XGBoost Regressor

Each model is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Best Model Saving
The model with the highest R² performance is automatically saved as:
```
best_model_<modelname>.pkl
```

## Feature Importance
Feature importance visualization is generated using the Random Forest model.

## Installation
Install required packages:
```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib openpyxl
```

## Running the Script
Run:
```
python your_script_name.py
```

