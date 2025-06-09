import numpy as np
import pandas as pd

# Read the preprocessed dataset
df = pd.read_csv("dataset/final/preprocessed_data_v2.csv")

# Check class balance
y = df['readmitted'] if 'readmitted' in df.columns else None
if y is not None:
    print("\n=== Class Balance (readmitted) ===")
    print(y.value_counts(normalize=True))

# Check for columns with a single unique value
print("\n=== Columns with a Single Unique Value ===")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"{col}: {df[col].unique()[0]}")

# Check for highly correlated features
print("\n=== Highly Correlated Features (|corr| > 0.95) ===")
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
print(high_corr)

# Check numerical feature distributions
print("\n=== Numerical Feature Distributions ===")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'readmitted' in numerical_cols:
    numerical_cols.remove('readmitted')
print(df[numerical_cols].describe())

# Check for missing values
print("\n=== Missing Values After Preprocessing ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Basic information
print("\n=== Dataset Information ===")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("\nColumns and their data types:")
print(df.dtypes)

# Analyze categorical columns
print("\n=== Categorical Columns Analysis ===")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].value_counts().head())

# Analyze numerical columns
print("\n=== Numerical Columns Analysis ===")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical columns statistics:")
print(df[numerical_cols].describe())

# Check for potential outliers in numerical columns
print("\n=== Potential Outliers Analysis ===")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
    if len(outliers) > 0:
        print(f"\nPotential outliers in {col}: {len(outliers)} values") 