
# ================================================================
# Advanced ML pipeline for Tensile Strength:
#  - processing / structural features handling
#  - advanced feature engineering (interactions, nonlinear transforms, PCA)
#  - multiple model comparison + deep learning MLP regressor
#  - saves best model
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Conv1D, Flatten, MaxPooling1D
import xgboost as xgb
import joblib

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    tf_available = True
except Exception as e:
    tf_available = False
    print("TensorFlow not available. Deep learning model will be skipped. To enable it, install tensorflow.")

# ----------------------------
# 0. CONFIG
# ----------------------------
FILE_PATH = "X_bp.xlsx"
TARGET_COL = "Tensile Strength, Mpa"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PCA_VARIANCE_KEEP = 0.95  # keep 95% variance in PCA

# Candidate processing/structural columns to accept if present
optional_structural_cols = [
    "Curing Temperature, C",
    "Curing Time, min",
    "Processing Pressure, MPa",
    "Fiber Orientation (deg)",
    "Fiber Type",            # categorical
    "Void Fraction, %"
]

# ----------------------------
# 1. LOAD DATA
# ----------------------------
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Input file not found: {FILE_PATH}")

df = pd.read_excel(FILE_PATH)
print("Data Loaded Successfully!")
print(df.head())

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("Dropped 'Unnamed: 0' index column.")

print("\nDataset info:")
print(df.info())

# ----------------------------
# 2. BASIC CLEANING & STRUCTURE FEATURES
# ----------------------------
# Check target exists
if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in the dataset.")

# Show which optional processing columns exist
present_optional = [c for c in optional_structural_cols if c in df.columns]
missing_optional = [c for c in optional_structural_cols if c not in df.columns]
print(f"\nFound optional structural/processing columns: {present_optional}")
if missing_optional:
    print(f"Optional columns not present (you may consider adding them): {missing_optional}")

# Drop rows with missing target
initial_rows = df.shape[0]
df = df.dropna(subset=[TARGET_COL])
dropped_rows = initial_rows - df.shape[0]
if dropped_rows > 0:
    print(f"Dropped {dropped_rows} rows with missing target.")

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# If optional structural categorical like 'Fiber Type' present, keep as categorical
# Ensure present optional columns are included in either numeric or categorical sets
for c in present_optional:
    if c in X.columns and c not in numeric_cols and c not in categorical_cols:
        # Attempt to coerce numeric if it looks numeric
        try:
            X[c] = pd.to_numeric(X[c])
            if c not in numeric_cols:
                numeric_cols.append(c)
        except:
            if c not in categorical_cols:
                categorical_cols.append(c)

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# ----------------------------
# 3. IMPUTATION & LOW-VARIANCE FILTER
# ----------------------------
# Impute numeric with median, categorical with most frequent
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_num = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)
if categorical_cols:
    X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)
else:
    X_cat = pd.DataFrame(index=X_num.index)

# Combine back
X_clean = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

# Remove zero/near-zero variance numeric cols
selector = VarianceThreshold(threshold=1e-5)
selector.fit(X_clean.select_dtypes(include=[np.number]))
keep_num = X_clean.select_dtypes(include=[np.number]).columns[selector.get_support()]
removed = set(X_clean.select_dtypes(include=[np.number]).columns) - set(keep_num)
if removed:
    print(f"Removed near-zero variance numeric features: {removed}")
X_clean = X_clean[ list(keep_num) + list(X_clean.select_dtypes(exclude=[np.number]).columns) ]

# ----------------------------
# 4. FEATURE ENGINEERING
#  - nonlinear transforms
#  - interaction terms (polynomial interactions)
#  - PCA for dimensionality reduction
# ----------------------------
fe_df = X_clean.copy()

# Nonlinear transforms: apply log1p and sqrt to positive numeric features (avoid negative values)
numeric_now = fe_df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_now:
    col_min = fe_df[col].min()
    if col_min >= 0:
        fe_df[f"{col}_log1p"] = np.log1p(fe_df[col])
        fe_df[f"{col}_sqrt"] = np.sqrt(fe_df[col])
    else:
        # skip log for columns with negatives
        fe_df[f"{col}_sqrt_shift"] = np.sqrt(fe_df[col] - col_min + 1e-6)

# Interaction terms: use PolynomialFeatures (degree=2, interactions only)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_num = fe_df.select_dtypes(include=[np.number])

# ✅ First fit to data, then get feature names
poly_arr = poly.fit_transform(poly_num)
poly_names = poly.get_feature_names_out(poly_num.columns)

poly_df = pd.DataFrame(poly_arr, columns=poly_names, index=fe_df.index)

# To keep feature explosion manageable, drop original numeric columns and keep interactions + transforms
# but we will concatenate a limited set: all interaction terms but drop pure (single-feature) columns from polynomial output
interaction_cols = [c for c in poly_names if " " in c or ":" in c or "_" in c and (" " not in c)]
# NOTE: sklearn uses feature naming with separators; keep all columns with '*' or '^' depends on sklearn version.
# Simpler approach: take columns with names containing '^' or ' ' or ':'; else keep everything but limit dimensionality.
interaction_df = poly_df.loc[:, poly_df.columns.difference(numeric_now)]

# If interactions too many, keep top-K by variance
MAX_INTERACTIONS = 200
if interaction_df.shape[1] > MAX_INTERACTIONS:
    variances = interaction_df.var().sort_values(ascending=False)
    top_inter = variances.index[:MAX_INTERACTIONS].tolist()
    interaction_df = interaction_df[top_inter]
    print(f"Trimmed interaction features to top {MAX_INTERACTIONS} by variance.")

# Combine features
fe_combined = pd.concat([fe_df.reset_index(drop=True), interaction_df.reset_index(drop=True)], axis=1)

# One-hot encode small cardinality categorical cols (if any)
if categorical_cols:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cat_enc = pd.DataFrame(ohe.fit_transform(X_clean[categorical_cols]), columns=ohe.get_feature_names_out(categorical_cols), index=fe_combined.index)
    fe_combined = pd.concat([fe_combined.reset_index(drop=True), cat_enc.reset_index(drop=True)], axis=1)

# Standardize numeric features before PCA / modeling
scaler = StandardScaler()
numeric_feats = fe_combined.select_dtypes(include=[np.number]).columns.tolist()
fe_scaled = pd.DataFrame(scaler.fit_transform(fe_combined[numeric_feats]), columns=numeric_feats, index=fe_combined.index)

# PCA to reduce dimensionality while keeping PERCENT variance
pca = PCA(n_components=PCA_VARIANCE_KEEP, svd_solver="full")
fe_pca = pca.fit_transform(fe_scaled)
fe_pca_df = pd.DataFrame(fe_pca, columns=[f"PCA_{i+1}" for i in range(fe_pca.shape[1])], index=fe_combined.index)
print(f"PCA reduced numeric features from {len(numeric_feats)} to {fe_pca.shape[1]} components (keep {PCA_VARIANCE_KEEP*100:.0f}% variance).")

# Final feature matrix: PCA components + any non-numeric columns (should be none after encoding)
X_final = fe_pca_df.copy()

# ----------------------------
# 5. TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ----------------------------
# 6. Prepare models
# ----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(C=10, kernel="rbf"),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "XGBoost Regressor": xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                          subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE)
}


if tf_available:
    # We'll build the Keras model later inside training loop to ensure reproducibility
    models["DeepNN"] = "Keras_MLP_placeholder"
    models["CNN"] = "Keras_CNN_placeholder"

# ----------------------------
# 7. Train, evaluate and cross-validate
# ----------------------------
results = []
best_model = None
best_score = -np.inf
best_name = None

def evaluate_and_store(name, model_obj, Xtr, Xte, ytr, yte):
    # Train
    model_obj.fit(Xtr, ytr)
    preds = model_obj.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)
    # CV R2 (use KFold)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_scores = cross_val_score(model_obj, X_final, y, cv=cv, scoring="r2")
        cv_mean = np.mean(cv_scores)
    except Exception:
        cv_scores = None
        cv_mean = np.nan
    results.append([name, mae, rmse, r2, cv_scores, cv_mean])
    return mae, rmse, r2

# Train classical models
for name, obj in list(models.items()):
    if name in ["DeepNN", "CNN"]:
        continue
    print(f"\nTraining {name} ...")
    mae, rmse, r2 = evaluate_and_store(name, obj, X_train, X_test, y_train, y_test)
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    if r2 > best_score:
        best_score = r2
        best_model = obj
        best_name = name

# Train Deep NN
nn_history = None
if tf_available:
    print("\nTraining Deep Neural Network (MLP) ...")
    tf.random.set_seed(RANDOM_STATE)
    n_inputs = X_train.shape[1]
    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_inputs,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=32,
                        callbacks=[es], verbose=0)
    nn_history = history
    # Predict & evaluate
    preds_nn = model.predict(X_test).ravel()
    mae_nn = mean_absolute_error(y_test, preds_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, preds_nn))
    r2_nn = r2_score(y_test, preds_nn)
    print(f"DeepNN -> MAE: {mae_nn:.3f}, RMSE: {rmse_nn:.3f}, R2: {r2_nn:.3f}")
    results.append(["DeepNN", mae_nn, rmse_nn, r2_nn, None, None])
    if r2_nn > best_score:
        best_score = r2_nn
        best_model = model
        best_name = "DeepNN"


# Train Convolutional Neural Network (1D CNN)
cnn_history = None
if tf_available:
    print("\nTraining Convolutional Neural Network (1D CNN) ...")

    # reshape input for CNN: (samples, features, 1)
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)

    tf.random.set_seed(RANDOM_STATE)
    n_features = X_train.shape[1]

    cnn_model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(n_features, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="linear")
    ])
    cnn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es_cnn = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train,
        validation_split=0.1,
        epochs=200,
        batch_size=32,
        callbacks=[es_cnn],
        verbose=0
    )

    preds_cnn = cnn_model.predict(X_test_cnn).ravel()
    mae_cnn = mean_absolute_error(y_test, preds_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(y_test, preds_cnn))
    r2_cnn = r2_score(y_test, preds_cnn)

    print(f"CNN -> MAE: {mae_cnn:.3f}, RMSE: {rmse_cnn:.3f}, R2: {r2_cnn:.3f}")
    results.append(["CNN", mae_cnn, rmse_cnn, r2_cnn, None, None])

    if r2_cnn > best_score:
        best_score = r2_cnn
        best_model = cnn_model
        best_name = "CNN"

# ----------------------------
# 8. Results summary
# ----------------------------
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "CV_Scores", "CV_Mean"])
results_df_sorted = results_df.sort_values(by="R2", ascending=False)
print("\nMODEL PERFORMANCE (sorted by R2):")
print(results_df_sorted[["Model", "MAE", "RMSE", "R2", "CV_Mean"]])

# ----------------------------
# 9. Plots
# ----------------------------
# Correlation heatmap (original df)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="viridis")
plt.title("Correlation heatmap (original features)")
plt.tight_layout()
plt.show()

# Feature importance from Random Forest if present
if "Random Forest" in models:
    rf = models["Random Forest"]
    try:
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        # Map back to PCA components
        feat_names = X_final.columns
        feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=False).head(20)
        plt.figure(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp)
        plt.title("Top 20 Feature Importances (Random Forest on PCA features)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not compute RandomForest feature importances:", e)

# NN training history plots
if nn_history is not None:
    plt.figure(figsize=(8,4))
    plt.plot(nn_history.history["loss"], label="train_loss")
    plt.plot(nn_history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("NN training and validation loss")
    plt.tight_layout()
    plt.show()

# CNN training history plot
if cnn_history is not None:
    plt.figure(figsize=(8,4))
    plt.plot(cnn_history.history["loss"], label="train_loss")
    plt.plot(cnn_history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("CNN training and validation loss")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 10. Save best model
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if best_model is not None:
    save_name = f"best_model_{best_name}_{timestamp}.pkl"
    # If best_model is Keras model, save differently
    if tf_available and best_name == "DeepNN" and isinstance(best_model, tf.keras.Model):
        best_model.save(f"best_model_DeepNN_{timestamp}")
        print(f"Saved Keras model to folder: best_model_DeepNN_{timestamp}")
    else:
        joblib.dump(best_model, save_name)
        print(f"Saved best model as: {save_name}")
else:
    print("No best model found to save.")

