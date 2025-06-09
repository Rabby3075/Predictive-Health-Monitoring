import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import os

# Load test set
X_test = pd.read_csv('models/X_test.csv')
y_test = pd.read_csv('models/y_test.csv')

# List of models to evaluate
model_files = [
    ('Logistic Regression', 'models/logistic_regression.joblib'),
    ('Random Forest', 'models/random_forest.joblib'),
    ('MLP Classifier', 'models/mlp_classifier.joblib'),
    ('XGBoost', 'models/xgboost.joblib')
]

os.makedirs('report/eval_plots', exist_ok=True)

for name, path in model_files:
    if os.path.exists(path):
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else y_pred
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'report/eval_plots/{name.replace(" ", "_").lower()}_confusion_matrix.png')
        plt.close()
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f'report/eval_plots/{name.replace(" ", "_").lower()}_roc_curve.png')
        plt.close()
        # Feature Importance (for tree-based models)
        if name in ['Random Forest', 'XGBoost'] and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = importances.argsort()[::-1][:10]
            features = X_test.columns[indices]
            plt.figure(figsize=(8,6))
            sns.barplot(x=importances[indices], y=features)
            plt.title(f'{name} Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(f'report/eval_plots/{name.replace(" ", "_").lower()}_feature_importance.png')
            plt.close()
        print(f'Plots saved for {name}')
    else:
        print(f'Model file not found: {path}')
print('All evaluation plots saved in report/eval_plots/') 