import os

import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/raw/hospital_readmissions.csv")

# Filter for aged care relevant patients (60+)
df = df[df['age'].isin(['[60-70)', '[70-80)', '[80-90)', '[90-100)'])]

# Convert age ranges to numerical midpoints
age_map = {
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}
df['age'] = df['age'].map(age_map)

# Handle missing medical specialties
df['medical_specialty'] = df['medical_specialty'].replace('Missing', 'Unknown')

# Create combined diagnosis feature
def get_primary_diagnosis(row):
    diagnoses = [row['diag_1'], row['diag_2'], row['diag_3']]
    # Count occurrences of each diagnosis
    diag_counts = pd.Series(diagnoses).value_counts()
    # Return the most frequent diagnosis
    return diag_counts.index[0] if not diag_counts.empty else 'Other'

df['primary_diagnosis'] = df.apply(get_primary_diagnosis, axis=1)

# Handle outliers in numerical columns using IQR method
numerical_cols = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 
                 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)

# Convert target variable 'readmitted' to binary
df['readmitted'] = df['readmitted'].map({'no': 0, 'yes': 1})

# Encode categorical columns
categorical_cols = ['medical_specialty', 'diag_1', 'diag_2', 'diag_3', 
                   'glucose_test', 'A1Ctest', 'change', 'diabetes_med', 
                   'primary_diagnosis']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Remove missing values
df_encoded = df_encoded.dropna()

# Create the output directory if it doesn't exist
os.makedirs('dataset/final', exist_ok=True)

# Save the preprocessed dataset with a new filename
output_path = 'dataset/final/preprocessed_data_v2.csv'
df_encoded.to_csv(output_path, index=False)

print(f"Preprocessed data has been saved to: {output_path}")
print(f"Shape of the preprocessed dataset: {df_encoded.shape}")
