"""
Sri Lanka Bank Customer Churn — LightGBM Training Script
=========================================================
Trains on research-based Sri Lankan data (generate_sl_data.py output).
Features: Age, Tenure_Years, Account_Balance_LKR, District,
          CRIB_Status, Recent_Loan_Rejected, Digital_Banking_User
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("📂 Loading data...")
df = pd.read_csv('sl_bank_data.csv')
print(f"   Rows: {len(df)}, Churn rate: {df['Churn'].mean()*100:.1f}%")

FEATURES = ['Age', 'Tenure_Years', 'Account_Balance_LKR', 'District',
            'CRIB_Status', 'Recent_Loan_Rejected', 'Digital_Banking_User']
TARGET = 'Churn'

X = df[FEATURES]
y = df[TARGET]

# =============================================================================
# 2. TRAIN / TEST SPLIT (stratified to preserve churn ratio)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# =============================================================================
# 3. PREPROCESSOR
#    - Numeric: StandardScaler (Age, Tenure, Balance)
#    - Categorical: OneHotEncoder (District, CRIB_Status)
#    - Binary: pass-through (Recent_Loan_Rejected, Digital_Banking_User)
# =============================================================================
numeric_features     = ['Age', 'Tenure_Years', 'Account_Balance_LKR']
categorical_features = ['District', 'CRIB_Status']
binary_features      = ['Recent_Loan_Rejected', 'Digital_Banking_User']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
    ('bin', 'passthrough', binary_features)
])

# Fit preprocessor
print("\n⚙️  Fitting preprocessor...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Convert sparse to dense if needed
import scipy.sparse
if scipy.sparse.issparse(X_train_proc):
    X_train_proc = X_train_proc.toarray()
if scipy.sparse.issparse(X_test_proc):
    X_test_proc  = X_test_proc.toarray()

feature_names = preprocessor.get_feature_names_out()
print(f"   Features after encoding: {len(feature_names)}")

X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names)

# =============================================================================
# 4. COMPUTE CLASS WEIGHTS
#    Handle imbalanced churn (typically minority class)
# =============================================================================
churn_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n⚖️  Class weight ratio (non-churn/churn): {churn_ratio:.2f}")

# =============================================================================
# 5. LIGHTGBM MODEL — Tuned for Sri Lankan banking data
# =============================================================================
model = LGBMClassifier(
    n_estimators       = 500,
    learning_rate      = 0.05,
    max_depth          = 6,
    num_leaves         = 40,
    min_child_samples  = 20,
    subsample          = 0.80,
    colsample_bytree   = 0.80,
    reg_alpha          = 0.1,
    reg_lambda         = 0.1,
    scale_pos_weight   = churn_ratio,  # Handle class imbalance
    random_state       = 42,
    n_jobs             = -1,
    verbose            = -1
)

print("\n🚀 Training LightGBM model...")
model.fit(
    X_train_df, y_train,
    eval_set=[(X_test_df, y_test)],
    callbacks=[
        __import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False),
        __import__('lightgbm').log_evaluation(period=100)
    ]
)

# =============================================================================
# 6. EVALUATION
# =============================================================================
y_pred      = model.predict(X_test_df)
y_pred_prob = model.predict_proba(X_test_df)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
cm      = confusion_matrix(y_test, y_pred)

print(f"\n{'='*55}")
print(f"  🏆 RESULTS — Sri Lankan Banking Churn Model")
print(f"{'='*55}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  {'':10} Predicted No  Predicted Yes")
print(f"  Actual No  {cm[0][0]:>12}  {cm[0][1]:>12}")
print(f"  Actual Yes {cm[1][0]:>12}  {cm[1][1]:>12}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance (top 10)
fi = pd.Series(model.feature_importances_, index=feature_names)
print(f"\n  🔑 Top 10 Feature Importances:")
print(fi.sort_values(ascending=False).head(10).to_string())

# =============================================================================
# 7. SAVE MODEL AND PREPROCESSOR
# =============================================================================
print(f"\n💾 Saving model and preprocessor...")
joblib.dump(model,        'sl_model.joblib')
joblib.dump(preprocessor, 'sl_preprocessor.joblib')
print(f"  ✅ sl_model.joblib saved")
print(f"  ✅ sl_preprocessor.joblib saved")
print(f"\n🎉 Training complete! Sri Lankan banking churn model ready.")
