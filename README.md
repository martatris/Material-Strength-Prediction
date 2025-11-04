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

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- (optional) gseapy

To install all dependencies:
    pip install numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm gseapy

--------------------------------------------------------------
Input Data - Downloading the Dataset
--------------------------------------------------------------
Dataset: **GSE30550 - Influenza A virus infection of human plasmacytoid dendritic cells (pDCs)**

You can obtain the data in two ways:

**Option 1: Manual Download**
1. Go to the NCBI GEO dataset page:  
   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30550  
2. Scroll to the “Download family” section.  
3. Download the file named **GSE30550_series_matrix.txt.gz**
4. Extract the file (e.g., using WinRAR, 7-Zip, or `gunzip` on macOS/Linux):
   gunzip GSE30550_series_matrix.txt.gz
5. Place the extracted file (`GSE30550_series_matrix.txt`) in the same directory 
   as the Python script (`Influenza.py`).

**Option 2: Command Line Download (macOS/Linux)**
If you have `wget` installed:
   wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30550/matrix/GSE30550_series_matrix.txt.gz
   gunzip GSE30550_series_matrix.txt.gz

Then confirm the file exists with:
   ls GSE30550_series_matrix.txt

--------------------------------------------------------------
How to Run
--------------------------------------------------------------
1. Make sure the dataset (`GSE30550_series_matrix.txt`) is located in the same folder.
2. Open a terminal in this directory and run:

    python Influenza.py

The script will:
- Load and preprocess the gene expression matrix
- Perform ANOVA feature selection (top 200 genes)
- Train and evaluate multiple classification models
- Print detailed evaluation metrics
- Display visualizations and summary tables

--------------------------------------------------------------
Models Evaluated
--------------------------------------------------------------
The script compares the following classifiers:

- Logistic Regression
- Support Vector Machine (RBF Kernel)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM

Each model is evaluated using 5-fold stratified cross-validation.

Metrics reported:
- Accuracy
- F1 Score (weighted)
- ROC-AUC (one-vs-rest)

--------------------------------------------------------------
Output and Visualization
--------------------------------------------------------------
The following visual outputs are generated:

- Confusion matrix for each model
- PCA scatter plot of top 200 ANOVA-selected genes
- Heatmap showing expression of selected genes
- Cross-validation summary table printed in the console

If GSEApy is installed, pathway enrichment analysis is also attempted.

--------------------------------------------------------------
Notes
--------------------------------------------------------------
- Some classifiers (like LightGBM or XGBoost) may require extra memory; 
  if you encounter issues, reduce the number of folds or disable parallel jobs.
- The dataset labels are inferred heuristically based on sample names 
  ("hour", "post", or "flu" → Infected).

--------------------------------------------------------------
Example Results
--------------------------------------------------------------
After running, you will see outputs similar to:

    Selected top 200 genes using ANOVA F-test
    Evaluating Random Forest...
    CV Accuracy: 0.94 | F1: 0.93 | ROC-AUC: 0.95
    Test Accuracy: 0.92 | F1: 0.91 | ROC-AUC: 0.94
    Classification Report:
                precision  recall  f1-score  support
      Control       0.90     0.93     0.91
      Infected      0.94     0.92     0.93


