import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the preprocessed dataset
df = pd.read_csv("dataset/final/preprocessed_data_v2.csv")

# Create output directory for plots
output_dir = 'report/eda_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

def safe_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

# 1. Target Variable Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='readmitted', data=df)
plt.title('Readmitted Distribution')
plt.savefig(f'{output_dir}/readmitted_distribution.png')
plt.close()

# 2. Numerical Feature Distributions
num_cols = ['age', 'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
for col in num_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'{col} Distribution')
        plt.savefig(f'{output_dir}/{safe_filename(col)}_hist.png')
        plt.close()
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'{col} Boxplot')
        plt.savefig(f'{output_dir}/{safe_filename(col)}_box.png')
        plt.close()

# 3. Categorical Feature Distributions (one-hot encoded columns)
cat_cols = [col for col in df.columns if df[col].dtype == 'bool']
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f'{col} Distribution')
    plt.savefig(f'{output_dir}/{safe_filename(col)}_bar.png')
    plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(12,10))
corr = df[[c for c in num_cols if c in df.columns] + ['readmitted']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(f'{output_dir}/correlation_heatmap.png')
plt.close()

# 5. Pairplot (for selected features)
plot_cols = [c for c in num_cols if c in df.columns] + ['readmitted']
sns.pairplot(df[plot_cols], hue='readmitted', diag_kind='kde')
plt.savefig(f'{output_dir}/pairplot.png')
plt.close()

# 6. Relationship between Features and Target (boxplots)
for col in num_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        plt.figure(figsize=(6,4))
        sns.boxplot(x='readmitted', y=col, data=df)
        plt.title(f'{col} vs Readmitted')
        plt.savefig(f'{output_dir}/{safe_filename(col)}_vs_readmitted_box.png')
        plt.close()

print('EDA visualizations saved in', output_dir) 