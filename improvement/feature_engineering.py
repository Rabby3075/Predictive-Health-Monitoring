import pandas as pd
import numpy as np
import os

df = pd.read_csv('dataset/final/preprocessed_data_v2.csv')

# Example interaction features
df['age_x_time_in_hospital'] = df['age'] * df['time_in_hospital']
df['medications_per_day'] = df['n_medications'] / (df['time_in_hospital'] + 1)

# Example aggregate features
df['total_visits'] = df['n_outpatient'] + df['n_inpatient'] + df['n_emergency']

# Example comorbidity score (count of diagnosis columns not 'Other' or 'Missing')
diag_cols = [col for col in df.columns if col.startswith('diag_')]
def comorbidity(row):
    return sum([row[c] not in ['Other', 'Missing'] for c in diag_cols])
df['comorbidity_score'] = df.apply(comorbidity, axis=1)

os.makedirs('improvement', exist_ok=True)
df.to_csv('improvement/engineered_data.csv', index=False)
print('Feature engineering complete. Saved to improvement/engineered_data.csv') 