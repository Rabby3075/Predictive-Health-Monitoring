import os

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Load test set
X_test = pd.read_csv('models/X_test.csv')
y_test = pd.read_csv('models/y_test.csv')

results = []

# List of models to evaluate
model_files = [
    ('Logistic Regression', 'models/logistic_regression.joblib'),
    ('Random Forest', 'models/random_forest.joblib'),
    ('MLP Classifier', 'models/mlp_classifier.joblib'),
    ('XGBoost', 'models/xgboost.joblib')
]

for name, path in model_files:
    if os.path.exists(path):
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else y_pred
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc})
        print(f"{name}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}")
    else:
        print(f"Model file not found: {path}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('report/model_evaluation_summary.csv', index=False)
print('Evaluation summary saved to report/model_evaluation_summary.csv') 