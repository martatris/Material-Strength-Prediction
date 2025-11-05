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

## Results Summary

The dataset contained 1023 composite material samples with 10 formulation-related variables. No processing or structural manufacturing parameters (such as curing temperature, fiber orientation, or void fraction) were available, which limits predictive capacity.

Advanced feature engineering was performed, including nonlinear interaction terms and PCA dimensionality reduction (retaining 95% of variance). Five regression models were evaluated:

| Model                     | MAE (MPa) | RMSE (MPa) | R² Score | CV Mean
|--------------------------|-----------|------------|----------|----------|
| Linear Regression         | 364.971    | 464.153     | 0.007    | -0.010359 |
| Support Vector Regressor  | 365.798    | 466.705     | -0.004   | -0.007211 |
| Random Forest Regressor   | 370.842    | 471.913     | -0.027   | -0.052908 |
| XGBoost Regressor         | 400.740    | 506.494     | -0.182   | -0.179062 |
| Deep Neural Network (MLP) | 530.818    | 715.990     | -1.363   | NaN       |

### Interpretation

The best-performing model was **Linear Regression**, though all models showed **low predictive power (R² ≈ 0)**. This indicates that the available formulation variables alone cannot explain the variation in tensile strength.

### Recommendation

To improve model accuracy, the dataset should be expanded to include key processing and microstructural parameters such as:
- Curing temperature and time
- Processing pressure
- Fiber orientation and type
- Void fraction (microstructure quality)

These are physically known to influence interfacial bonding and load transfer and are likely missing drivers of tensile strength variability.
