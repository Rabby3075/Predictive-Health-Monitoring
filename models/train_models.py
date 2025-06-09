import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Optional: import xgboost if available
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

# Load data
df = pd.read_csv('dataset/final/preprocessed_data_v2.csv')
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('models', exist_ok=True)

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, 'models/logistic_regression.joblib')

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'models/random_forest.joblib')

# 3. XGBoost (if available)
if xgb_installed:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, 'models/xgboost.joblib')

# 4. MLPClassifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
joblib.dump(mlp, 'models/mlp_classifier.joblib')

# Save test set for evaluation
X_test.to_csv('models/X_test.csv', index=False)
y_test.to_csv('models/y_test.csv', index=False)
print('All models trained and saved. Test set saved for evaluation.') 