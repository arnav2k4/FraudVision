import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import shap
import json

# Load Data
df = pd.read_csv("transactions_train.csv")  # Replace with actual dataset

# Convert transaction_date to datetime & extract features
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['hour_of_day'] = df['transaction_date'].dt.hour
df['day_of_week'] = df['transaction_date'].dt.dayofweek
df.drop(columns=['transaction_date'], inplace=True)

# Encode categorical features
categorical_cols = ['transaction_channel', 'transaction_payment_mode_anonymous', 'payment_gateway_bank_anonymous']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Reduce Influence of Highly Correlated Anonymous Features
anon_cols = ['payer_email_anonymous', 'payee_ip_anonymous', 'payer_mobile_anonymous', 'transaction_id_anonymous']
for col in anon_cols:
    df[col] = df[col].astype(str).map(df[col].value_counts())
    df[col] = np.log1p(df[col])  # Log transformation to reduce variance

# Drop payee_id_anonymous to reduce overfitting
df.drop(columns=['payee_id_anonymous'], inplace=True)

# Fill missing values
df.fillna(df.median(), inplace=True)

# Define Features & Target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Handle Imbalanced Data with SMOTETomek
smote_tomek = SMOTETomek(sampling_strategy=0.1, random_state=42)  # Keep more fraud cases
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# 🚀 Optimized XGBoost with Better Fraud Detection
xgb_model = xgb.XGBClassifier(
    n_estimators=150,  # Increase tree count
    max_depth=3,  # Allow slightly deeper trees
    min_child_weight=10,  # Reduce overfitting
    learning_rate=0.02,  # Slower learning
    reg_lambda=50,  # Moderate L2 regularization
    reg_alpha=25,  # Moderate L1 regularization
    scale_pos_weight=30,  #  Increased to give fraud cases higher importance
    subsample=0.7,  # Prevent overfitting
    colsample_bytree=0.7,  # Use only 70% of features
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# Fit Model
xgb_model.fit(X_train, y_train)

# Cross-Validation
cv_scores = cross_val_score(xgb_model, X_resampled, y_resampled, cv=5, scoring="roc_auc")
print(f"Cross-Validation AUC-ROC Scores: {cv_scores}")
print(f"Mean AUC-ROC: {np.mean(cv_scores):.4f}")

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test)
fraud_scores = xgb_model.predict_proba(X_test)[:, 1]
print("📊 XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb, zero_division=1))
print("🎯 XGBoost AUC-ROC:", roc_auc_score(y_test, fraud_scores))

# Save Model
joblib.dump(xgb_model, "xgboost_fraud7.pkl")

# ✅ Isolation Forest for Anomaly Detection
iforest = IsolationForest(n_estimators=200, contamination=0.01, max_samples=0.6, random_state=42)  # Adjusted contamination rate
iforest.fit(X_train[y_train == 0])

# Save Model
joblib.dump(iforest, "iforest_fraud7.pkl")

# Predict Fraud with XGBoost
xgb_pred = xgb_model.predict(X_test)

# Apply Isolation Forest on XGBoost's Non-Fraud Predictions
non_fraud_cases = X_test[xgb_pred == 0]
iforest_pred = iforest.predict(non_fraud_cases)

# Convert Isolation Forest Output (-1 = fraud, 1 = normal)
iforest_pred = [1 if x == -1 else 0 for x in iforest_pred]

# Merge XGBoost & iForest Predictions
final_pred = xgb_pred.copy()
final_pred[xgb_pred == 0] = iforest_pred  # Replace XGBoost non-fraud cases with iForest results

# Evaluate Combined Model
print("📊 Final Model Performance (XGBoost + iForest):")
print(classification_report(y_test, final_pred, zero_division=1))
print("🎯 AUC-ROC:", roc_auc_score(y_test, final_pred))

# Test 1: Compare Training vs. Test Accuracy
y_train_pred = xgb_model.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)

y_test_pred = xgb_model.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"✅ Training Accuracy: {train_accuracy:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# 🔥 Feature Importance Analysis Using SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Generate Fraud Detection Output for Transactions
fraud_results = []
for idx, row in X_test.iterrows():
    result = {
        "transaction_id": str(idx),
        "is_fraud": bool(final_pred[idx]),
        "fraud_source": "model" if xgb_pred[idx] == 1 else "rule" if iforest_pred[idx] == 1 else "none",
        "fraud_reason": "High fraud probability" if fraud_scores[idx] > 0.7 else "Suspicious transaction",
        "fraud_score": float(fraud_scores[idx])
    }
    fraud_results.append(result)

# Save JSON output to a file
with open("fraud_results.json", "w") as f:
    json.dump(fraud_results, f, indent=4)

print("✅ Fraud detection results saved to fraud_results.json")
