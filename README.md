# Material Property Prediction (Tensile Strength ML Project)
--------------------------------------------------------------

Overview
--------------------------------------------------------------
This project predicts Tensile Strength (MPa) of composite materials based on various physical and chemical features using a Random Forest Regression model.
The script:

1. Loads your dataset (data.csv)
2. Preprocesses and scales features
3. Trains a Random Forest model
4. Evaluates performance (MAE, RMSE, R²)
5. Displays feature importance
6. Optionally saves the trained model as 'tensile_strength_model.pkl'

--------------------------------------------------------------
Dataset Format
--------------------------------------------------------------
Your dataset (data.csv) must include the following columns:

| Column Name | Description |
|------------|-------------|
| **Matrix-to-filler ratio** | Ratio of matrix material to filler |
| **Density, kg/m^3** | Material density |
| **Elastic modulus, GPa** | Elastic modulus |
| **Harderner content, wt.%** | Harderner content by weight |
| **Epoxy group content, %** | Epoxy group percentage |
| **Flash point, in Celcius** | Flash point temperature |
| **Surface Density, g/m^3** | Surface density |
| **Tensile modulus, GPa** | Tensile modulus |
| **Tensile strength, MPa** | (Target column), strength to predict |
| **Resin consumption, g/m^2** | Resin consumption |


Save the file as data.csv in the same directory as the script.

--------------------------------------------------------------
How to Run
--------------------------------------------------------------

1. Install required Python packages: 
   pip install pandas numpy scikit-learn matplotlib seaborn joblib 
2. Place your dataset file (data.csv) in the same folder. 
3. Run the script:
   python material_strength_prediction.py

The script will:
* Display training progress and results
* Show a feature importance plot
* Save the model as tensile_strength_model.pkl

--------------------------------------------------------------
Outputs
--------------------------------------------------------------

* Metrics printed: MAE, RMSE, R²
* Feature importance bar chart
* Saved model file: tensile_strength_model.pkl

--------------------------------------------------------------
Tips
--------------------------------------------------------------
* Ensure all numeric columns contain valid numerical data (no strings or missing values).
* You can adjust the target_col variable to predict a different property.
* Try increasing n_estimators in the Random Forest for higher accuracy (with longer training time).

