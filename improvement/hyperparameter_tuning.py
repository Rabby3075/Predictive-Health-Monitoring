import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

# Load engineered data
df = pd.read_csv('improvement/engineered_data.csv')
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('improvement/tuned_models', exist_ok=True)

results = []

# Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.01, 0.1, 1, 10]}, cv=3, scoring='f1')
grid_lr.fit(X_train, y_train)
joblib.dump(grid_lr.best_estimator_, 'improvement/tuned_models/logistic_regression.joblib')
results.append({'Model': 'Logistic Regression', 'Best Params': grid_lr.best_params_, 'Best F1': grid_lr.best_score_})

# Random Forest
grid_rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}, cv=3, scoring='f1')
grid_rf.fit(X_train, y_train)
joblib.dump(grid_rf.best_estimator_, 'improvement/tuned_models/random_forest.joblib')
results.append({'Model': 'Random Forest', 'Best Params': grid_rf.best_params_, 'Best F1': grid_rf.best_score_})

# XGBoost
if xgb_installed:
    grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'n_estimators': [100, 200], 'max_depth': [3, 6]}, cv=3, scoring='f1')
    grid_xgb.fit(X_train, y_train)
    joblib.dump(grid_xgb.best_estimator_, 'improvement/tuned_models/xgboost.joblib')
    results.append({'Model': 'XGBoost', 'Best Params': grid_xgb.best_params_, 'Best F1': grid_xgb.best_score_})

# MLPClassifier
grid_mlp = GridSearchCV(MLPClassifier(max_iter=300), {'hidden_layer_sizes': [(64,), (64,32)], 'alpha': [0.0001, 0.001]}, cv=3, scoring='f1')
grid_mlp.fit(X_train, y_train)
joblib.dump(grid_mlp.best_estimator_, 'improvement/tuned_models/mlp_classifier.joblib')
results.append({'Model': 'MLP Classifier', 'Best Params': grid_mlp.best_params_, 'Best F1': grid_mlp.best_score_})

pd.DataFrame(results).to_csv('improvement/tuning_results.csv', index=False)
print('Hyperparameter tuning complete. Best models and results saved.') 