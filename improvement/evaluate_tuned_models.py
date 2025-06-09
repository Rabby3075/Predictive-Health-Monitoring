import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

# Load test set (use the same split as in tuning)
df = pd.read_csv('improvement/engineered_data.csv')
X = df.drop('readmitted', axis=1)
y = df['readmitted']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('improvement/eval_plots', exist_ok=True)

results = []
model_files = [
    ('Logistic Regression', 'improvement/tuned_models/logistic_regression.joblib'),
    ('Random Forest', 'improvement/tuned_models/random_forest.joblib'),
    ('MLP Classifier', 'improvement/tuned_models/mlp_classifier.joblib'),
    ('XGBoost', 'improvement/tuned_models/xgboost.joblib')
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
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'improvement/eval_plots/{name.replace(" ", "_").lower()}_confusion_matrix.png')
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
        plt.savefig(f'improvement/eval_plots/{name.replace(" ", "_").lower()}_roc_curve.png')
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
            plt.savefig(f'improvement/eval_plots/{name.replace(" ", "_").lower()}_feature_importance.png')
            plt.close()
        print(f'Plots and metrics saved for {name}')
    else:
        print(f'Model file not found: {path}')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('improvement/eval_plots/tuned_model_evaluation_summary.csv', index=False)
print('Evaluation summary saved to improvement/eval_plots/tuned_model_evaluation_summary.csv') 