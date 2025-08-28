# Install xgboost if needed
# !pip install xgboost

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import xgboost as xgb

# Make predictions with XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# 6. Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"===== {model_name} =====")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    auc_score = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC: {auc_score:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}\n")

# Evaluate all models
evaluate_model(y_test, y_pred_log, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# 7. Feature Importance (Random Forest)
importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[-10:]
plt.figure(figsize=(8,6))
plt.barh(range(len(indices_rf)), importances_rf[indices_rf], align='center')
plt.yticks(range(len(indices_rf)), [X.columns[i] for i in indices_rf])
plt.title('Top 10 Important Features - Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 7b. Feature Importance (XGBoost)
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-10:]
plt.figure(figsize=(8,6))
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center')
plt.yticks(range(len(indices_xgb)), [X.columns[i] for i in indices_xgb])
plt.title('Top 10 Important Features - XGBoost')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 8. Final Report (Answers to Questions)
print("""
1. Data cleaning was performed using imputation for missing values, handling outliers, and checking multicollinearity.
2. Fraud detection models tested: Logistic Regression, Random Forest, and XGBoost.
3. Variable selection was done using correlation analysis, feature importance, and domain knowledge.
4. Performance measured using Confusion Matrix, ROC-AUC, and PR-AUC.
5. Key fraud predictors identified via feature importance.
6. Factors align with real-world fraud patterns (e.g., unusual transaction amounts, location changes).
7. Recommended prevention: real-time monitoring, MFA, encryption, fraud-scoring APIs.
8. Effectiveness can be determined via reduced fraud rate, better recall, and A/B testing.
""")
