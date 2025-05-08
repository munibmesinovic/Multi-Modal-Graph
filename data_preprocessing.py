#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torchtuples')
get_ipython().system('pip install pycox')


# In[ ]:


import pandas as pd
from itertools import islice
import random
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from functools import reduce
import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import torch # For building the networks
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast

import torch.nn.functional as F
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.evaluation import EvalSurv

seed = 250
random.seed(seed)
np.random.seed(seed)

import seaborn as sn
sn.set_theme(style="white", palette="rocket_r")

from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


cd ..


# In[ ]:


cd 'content/gdrive/MyDrive/MC-MED'


# # VISITS

# In[ ]:


visits = pd.read_csv('visits.csv')


# In[ ]:


print(f"Number of unique patients: {visits['MRN'].nunique()}")

print(f"Number of unique visits: {visits['CSN'].nunique()}")

visits.isnull().sum() / len(visits)


# In[ ]:


# Remove those with missing race, ethnicity, triage measurement, length of stay in ED

visits = visits.dropna(subset=['Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2', 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS'])
visits = visits[['MRN', 'CSN', 'Age', 'Gender', 'Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2',
                 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo']]


# In[ ]:


print(f"Number of unique patients: {visits['MRN'].nunique()}")

print(f"Number of unique visits: {visits['CSN'].nunique()}")


# In[ ]:


one_hot = pd.get_dummies(visits['Race'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Race',axis = 1)
# Join the encoded df
visits = visits.join(one_hot)


# In[ ]:


one_hot = pd.get_dummies(visits['Gender'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Gender',axis = 1)
# Join the encoded df
visits = visits.join(one_hot)


# In[ ]:


one_hot = pd.get_dummies(visits['Ethnicity'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Ethnicity',axis = 1)
# Join the encoded df
visits = visits.join(one_hot, rsuffix='_ethnicity')


# In[ ]:


ed_dispo_mapping = {
    'Discharge': 0,
    'ICU': 1,
    'Inpatient': 2,
    'Observation': 3
}

# Apply ordinal encoding to the 'ED_dispo' column
visits['Outcome'] = visits['ED_dispo'].map(ed_dispo_mapping)

visits.dropna(subset=['Outcome'], inplace=True)

triage_acuity_mapping = {
    '1-Resuscitation': 5,
    '2-Emergent': 4,
    '3-Urgent': 3,
    '4-Semi-Urgent': 2,
    '5-Non-Urgent': 1
}

visits['Triage_acuity_ordinal'] = visits['Triage_acuity'].map(triage_acuity_mapping)
visits['Triage_acuity_ordinal'].fillna(-1, inplace=True)

visits = visits.drop(['ED_dispo', 'Triage_acuity'],axis = 1)


# In[ ]:


# Calculate the distribution of 'Age'
age_distribution = visits['Age'].describe()
print("Age Distribution:\n", age_distribution)

visits = visits[visits['ED_LOS'] <= 24]

# Calculate the distribution of 'ED_LOS'
ed_los_distribution = visits['ED_LOS'].describe()
print("\nED_LOS Distribution:\n", ed_los_distribution)

# You can also visualize the distributions using histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(visits['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
plt.hist(visits['ED_LOS'], bins=20)
plt.xlabel('ED_LOS')
plt.ylabel('Frequency')
plt.title('Distribution of ED_LOS')

plt.tight_layout()
plt.show()


# In[ ]:


print(visits.Outcome.value_counts())


# # X-Ray Reports

# In[ ]:


rads = pd.read_csv('rads.csv')


# In[ ]:


rads.set_index('CSN', inplace=True)


# In[ ]:


# Drop rows with missing values in 'Study' or 'Impression'
rads = rads.dropna(subset=['Study', 'Impression'])


# In[ ]:


rads['Text'] = rads['Study'] + ' ' + rads['Impression']


# In[ ]:


rads.drop(['Study', 'Impression', 'Order_time', 'Result_time'], axis=1, inplace=True)


# In[ ]:


from transformers import AutoTokenizer, AutoModel

# Load Clinical-Longformer model
tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer").to("cuda")
model.eval()  # Set model to evaluation mode

# Function to get embeddings efficiently using GPU
def get_bert_embeddings(texts, batch_size=32, max_length=1024):
    """Extracts mean-pooled embeddings using GPU acceleration."""
    embeddings = []

    with torch.no_grad():  # Disable gradient computation
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Move tokenized input to GPU
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True,
                               padding=True, max_length=max_length).to("cuda")

            with autocast():  # Use mixed precision for speedup
                outputs = model(**inputs)  # Compute embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            embeddings.append(batch_embeddings.cpu().numpy())  # Move to CPU before stacking

    return np.vstack(embeddings)  # Stack outputs

# Extract features and maintain original patient index
X_bert_array = get_bert_embeddings(rads['Text'].tolist(), batch_size=32, max_length=1024)

# Convert to DataFrame while keeping patient index
X_bert_df = pd.DataFrame(X_bert_array, index=rads.index)

# Rename columns for clarity
X_bert_df.columns = [f'feat_{i}' for i in range(X_bert_df.shape[1])]

# Merge with original DataFrame if needed
rads_with_embeddings = pd.concat([rads, X_bert_df], axis=1)


# In[ ]:


rads_with_embeddings2 = rads_with_embeddings.copy()


# In[ ]:


rads_with_embeddings2.drop(['Text'], axis=1, inplace=True)


# In[ ]:


rads_with_embeddings2


# In[ ]:


visits.set_index('CSN', inplace=True)
rads_with_embeddings2.set_index('CSN', inplace=True)


# In[ ]:


visits = visits.merge(rads_with_embeddings2, how='left', left_index=True, right_index=True)
visits = visits.fillna(0)


# In[ ]:


visits.reset_index(inplace=True)


# In[ ]:





# # PAST MEDICAL HISTORY

# In[ ]:


pmh = pd.read_csv('pmh.csv')


# In[ ]:


pmh = pmh[['MRN', 'Code']]


# In[ ]:


# Identify MRNs present in the 'visits' DataFrame
mrns_in_visits = visits['MRN'].unique()

# Filter the 'pmh' DataFrame to keep only rows where 'MRN' is in 'mrns_in_visits'
pmh = pmh[pmh['MRN'].isin(mrns_in_visits)]


# In[ ]:


pmh.set_index('MRN', inplace=True)


# In[ ]:


# Get the top 500 most frequent codes
top_100_codes = pmh['Code'].value_counts().nlargest(500).index.tolist()

# Filter the pmh DataFrame to keep only rows with codes in the top 500
pmh = pmh[pmh['Code'].isin(top_100_codes)]


# In[ ]:


# One-hot encode ICD codes
icd_one_hot = pd.get_dummies(pmh['Code'])

# Sum over patients to get patient-ICD matrix
patient_icd_matrix = icd_one_hot.groupby('MRN').sum()


# In[ ]:


patient_icd_matrix


# In[ ]:


visits.set_index('MRN', inplace=True)


# In[ ]:


visits = visits.merge(patient_icd_matrix, how='left', left_index=True, right_index=True)
visits = visits.fillna(0)


# In[ ]:


visits.reset_index(inplace=True)


# In[ ]:


visits


# In[ ]:





# In[ ]:





# In[ ]:





# # Labels

# In[ ]:


AMI_patients = pd.read_csv('benchmark/new_acutemi/labeled_patients.csv')
PANCAN_patients = pd.read_csv('benchmark/new_pancan/labeled_patients.csv')
LUPUS_patients = pd.read_csv('benchmark/new_lupus/labeled_patients.csv')
HYPERTENSION_patients = pd.read_csv('benchmark/new_hypertension/labeled_patients.csv')
HYPERLIPIDEMIA_patients = pd.read_csv('benchmark/new_hyperlipidemia/labeled_patients.csv')
CELIAC_patients = pd.read_csv('benchmark/new_celiac/labeled_patients.csv')


# In[ ]:


AMI_patients = AMI_patients[['patient_id', 'prediction_time', 'value']]
AMI_patients = AMI_patients.rename(columns={'value': 'AMI'})
AMI_patients = AMI_patients.rename(columns={'prediction_time': 'AMI_Time'})
AMI_patients = AMI_patients.sort_values(by=['patient_id', 'AMI_Time']).groupby(['patient_id'])
AMI_patients = AMI_patients.last().sort_index().reset_index().drop_duplicates()

PANCAN_patients = PANCAN_patients[['patient_id', 'prediction_time', 'value']]
PANCAN_patients = PANCAN_patients.rename(columns={'value': 'PANCAN'})
PANCAN_patients = PANCAN_patients.rename(columns={'prediction_time': 'PANCAN_Time'})
PANCAN_patients = PANCAN_patients.sort_values(by=['patient_id', 'PANCAN_Time']).groupby(['patient_id'])
PANCAN_patients = PANCAN_patients.last().sort_index().reset_index().drop_duplicates()

LUPUS_patients = LUPUS_patients[['patient_id', 'prediction_time', 'value']]
LUPUS_patients = LUPUS_patients.rename(columns={'value': 'LUPUS'})
LUPUS_patients = LUPUS_patients.rename(columns={'prediction_time': 'LUPUS_Time'})
LUPUS_patients = LUPUS_patients.sort_values(by=['patient_id', 'LUPUS_Time']).groupby(['patient_id'])
LUPUS_patients = LUPUS_patients.last().sort_index().reset_index().drop_duplicates()

HYPERTENSION_patients = HYPERTENSION_patients[['patient_id', 'prediction_time', 'value']]
HYPERTENSION_patients = HYPERTENSION_patients.rename(columns={'value': 'HYPERTENSION'})
HYPERTENSION_patients = HYPERTENSION_patients.rename(columns={'prediction_time': 'HYPERTENSION_Time'})
HYPERTENSION_patients = HYPERTENSION_patients.sort_values(by=['patient_id', 'HYPERTENSION_Time']).groupby(['patient_id'])
HYPERTENSION_patients = HYPERTENSION_patients.last().sort_index().reset_index().drop_duplicates()

HYPERLIPIDEMIA_patients = HYPERLIPIDEMIA_patients[['patient_id', 'prediction_time', 'value']]
HYPERLIPIDEMIA_patients = HYPERLIPIDEMIA_patients.rename(columns={'value': 'HYPERLIPIDEMIA'})
HYPERLIPIDEMIA_patients = HYPERLIPIDEMIA_patients.rename(columns={'prediction_time': 'HYPERLIPIDEMIA_Time'})
HYPERLIPIDEMIA_patients = HYPERLIPIDEMIA_patients.sort_values(by=['patient_id', 'HYPERLIPIDEMIA_Time']).groupby(['patient_id'])
HYPERLIPIDEMIA_patients = HYPERLIPIDEMIA_patients.last().sort_index().reset_index().drop_duplicates()

CELIAC_patients = CELIAC_patients[['patient_id', 'prediction_time', 'value']]
CELIAC_patients = CELIAC_patients.rename(columns={'value': 'CELIAC'})
CELIAC_patients = CELIAC_patients.rename(columns={'prediction_time': 'CELIAC_Time'})
CELIAC_patients = CELIAC_patients.sort_values(by=['patient_id', 'CELIAC_Time']).groupby(['patient_id'])
CELIAC_patients = CELIAC_patients.last().sort_index().reset_index().drop_duplicates()

# AMI_patients= AMI_patients.loc[AMI_patients['AMI']==True]


# In[ ]:


AMI_patients['AMI'].value_counts(), PANCAN_patients['PANCAN'].value_counts(), LUPUS_patients['LUPUS'].value_counts(), HYPERTENSION_patients['HYPERTENSION'].value_counts(), HYPERLIPIDEMIA_patients['HYPERLIPIDEMIA'].value_counts(), CELIAC_patients['CELIAC'].value_counts()


# In[ ]:


final = pd.merge(AMI_patients, PANCAN_patients, left_on='patient_id', right_on='patient_id', how='left')
final = pd.merge(final, LUPUS_patients, left_on='patient_id', right_on='patient_id', how='left')
final = pd.merge(final, HYPERTENSION_patients, left_on='patient_id', right_on='patient_id', how='left')
final = pd.merge(final, HYPERLIPIDEMIA_patients, left_on='patient_id', right_on='patient_id', how='left')
final = pd.merge(final, CELIAC_patients, left_on='patient_id', right_on='patient_id', how='left')


# In[ ]:


final.isnull().sum(axis = 0)


# In[ ]:


final = pd.merge(final, start_time_for_labels[['patient_id', 'start']], left_on='patient_id', right_on='patient_id', how='left')


# In[ ]:


final['AMI'].fillna(False, inplace=True)
final['PANCAN'].fillna(False, inplace=True)
final['LUPUS'].fillna(False, inplace=True)
final['HYPERTENSION'].fillna(False, inplace=True)
final['HYPERLIPIDEMIA'].fillna(False, inplace=True)
final['CELIAC'].fillna(False, inplace=True)

final['AMI_Time'].fillna(final['start'], inplace=True)
final['PANCAN_Time'].fillna(final['start'], inplace=True)
final['LUPUS_Time'].fillna(final['start'], inplace=True)
final['HYPERTENSION_Time'].fillna(final['start'], inplace=True)
final['HYPERLIPIDEMIA_Time'].fillna(final['start'], inplace=True)
final['CELIAC_Time'].fillna(final['start'], inplace=True)


# In[ ]:


# final.dropna(inplace=True)


# In[ ]:


final['AMI_Time'] = pd.to_datetime(final['AMI_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['PANCAN_Time'] = pd.to_datetime(final['PANCAN_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['LUPUS_Time'] = pd.to_datetime(final['LUPUS_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['HYPERTENSION_Time'] = pd.to_datetime(final['HYPERTENSION_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['HYPERLIPIDEMIA_Time'] = pd.to_datetime(final['HYPERLIPIDEMIA_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['CELIAC_Time'] = pd.to_datetime(final['CELIAC_Time'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required
final['start'] = pd.to_datetime(final['start'], format='%Y-%m-%d %H:%M:%S', errors='ignore') #if conversion required


# In[ ]:


final['offset_AMI'] = (final['AMI_Time'] - final['start']).dt.days
final['offset_PANCAN'] = (final['PANCAN_Time'] - final['start']).dt.days
final['offset_LUPUS'] = (final['LUPUS_Time'] - final['start']).dt.days
final['offset_HYPERTENSION'] = (final['HYPERTENSION_Time'] - final['start']).dt.days
final['offset_HYPERLIPIDEMIA'] = (final['HYPERLIPIDEMIA_Time'] - final['start']).dt.days
final['offset_CELIAC'] = (final['CELIAC_Time'] - final['start']).dt.days


# In[ ]:


final = final.loc[(final['offset_AMI'] <= 365) & (final['offset_AMI'] > -1) & (final['offset_PANCAN'] <= 365) & (final['offset_PANCAN'] > -1)
& (final['offset_LUPUS'] <= 365) & (final['offset_LUPUS'] > -1)
& (final['offset_HYPERTENSION'] <= 365) & (final['offset_HYPERTENSION'] > -1)
& (final['offset_HYPERLIPIDEMIA'] <= 365) & (final['offset_HYPERLIPIDEMIA'] > -1)
& (final['offset_CELIAC'] <= 365) & (final['offset_CELIAC'] > -1)]


# In[ ]:


final = final.sort_values(by=['patient_id', 'start']).groupby(['patient_id']).first()


# In[ ]:


final = final.drop(['AMI_Time', 'PANCAN_Time', 'LUPUS_Time', 'HYPERTENSION_Time', 'HYPERLIPIDEMIA_Time', 'CELIAC_Time'], axis=1)


# In[ ]:


final.replace({False: 0, True: 1}, inplace=True)


# In[ ]:


final


# In[ ]:


final.to_csv('labels_with_start_time.csv', index=True)


# # Demographics

# In[ ]:


pd.reset_option('^display.', silent=True)
demographics = data.loc[data['omop_table'] == 'person']
# demographics.drop_duplicates(subset=['patient_id'], inplace=True)


# In[ ]:


age = demographics.loc[demographics['code'] == 'SNOMED/3950001']
age = age[['patient_id', 'start']]
age['start'] = pd.DatetimeIndex(age['start']).year
age = age.rename(columns={'start': 'birth_year'})


# In[ ]:


sex_women = demographics.loc[demographics['code'] == 'Gender/F']
sex_men = demographics.loc[demographics['code'] == 'Gender/M']
sex = pd.concat([sex_men, sex_women], axis=0, ignore_index=True)[['patient_id', 'code']]
sex = sex.rename(columns={'code': 'sex'})
sex['sex'] = sex['sex'].map({'Gender/F': 0, 'Gender/M': 1})


# In[ ]:


race_white = demographics.loc[demographics['code'] == 'Race/5']
race_asian = demographics.loc[demographics['code'] == 'Race/2']
race_black = demographics.loc[demographics['code'] == 'Race/3']
race_native = demographics.loc[demographics['code'] == 'Race/4']
race = pd.concat([race_white, race_asian, race_black, race_native], axis=0, ignore_index=True)[['patient_id', 'code']]
race = race.rename(columns={'code': 'race'})
one_hot = pd.get_dummies(race['race'], dtype=int)
# Drop column B as it is now encoded
race = race.drop('race',axis = 1)
# Join the encoded df
race = race.join(one_hot)
race = race.rename(columns={'Race/2': 'Asian', 'Race/5': 'White', 'Race/3': 'Black', 'Race/4': 'Native'})


# In[ ]:


demographics = pd.merge(age, sex, left_on='patient_id', right_on='patient_id', how='left')
demographics = pd.merge(demographics, race, left_on='patient_id', right_on='patient_id', how='left')
demographics.loc[demographics.Asian.isna(), 'Unknown_Race'] = 1
demographics = demographics.fillna(0)
demographics.drop_duplicates(subset=['patient_id'], inplace=True)
demographics.set_index(['patient_id'], inplace=True)
del age, sex, race, one_hot, sex_women, sex_men, race_white, race_asian, race_black, race_native


# In[ ]:


demographics


# In[ ]:


demographics.to_csv('demographics.csv', index=True)


# In[ ]:





# # Labs

# In[ ]:


time_series_labs = pd.read_csv('labs.csv')


# In[ ]:


# Convert Result_time to datetime, handling out-of-bounds dates
time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')

# Remove rows with NaT (Not a Time) in Result_time, which were created due to out-of-bounds dates
time_series_labs = time_series_labs[time_series_labs['Result_time'].notna()]

# Format Result_time as 'YYYY-MM-DD HH:MM:SS'
time_series_labs['Result_time'] = time_series_labs['Result_time'].dt.strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


# Pivot the DataFrame
time_series_labs = time_series_labs.pivot_table(
    index=['CSN', 'Result_time'],  # Index: CSN and Result_time
    columns='Component_name',      # Columns: Unique Component_name entries
    values='Component_value',      # Values: Component_value (or another column like Component_result)
    aggfunc='first'                # Handle duplicates: Take the first occurrence
)


# In[ ]:


# Keep columns with less than 90% missing values
threshold = 0.85
time_series_labs = time_series_labs.loc[:, time_series_labs.isnull().sum() / len(time_series_labs) < threshold]


# In[ ]:


# Convert problematic columns to numeric before calculating the mean
for column in time_series_labs.columns:
    if column != 'CSN' and column != 'Result_time':  # Exclude non-numeric columns if present
        time_series_labs[column] = pd.to_numeric(time_series_labs[column], errors='coerce')


# In[ ]:


time_series_labs.reset_index(inplace=True)
time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')
time_series_labs.set_index(['CSN', 'Result_time'], inplace=True)
# Time shift so that the starting point for each sample is time = 0
time_series_labs.reset_index(level=1, inplace=True)
minimum_shifts = time_series_labs.groupby('CSN')['Result_time'].min()
# minimum_shifts = timeseries_summary.time.min(level=0)
time_series_labs = time_series_labs.merge(minimum_shifts, left_index=True, right_index=True)
time_series_labs['Result_time'] = time_series_labs['Result_time_x'] - time_series_labs['Result_time_y']
time_series_labs.drop(columns=['Result_time_x', 'Result_time_y'], inplace=True)
time_series_labs.set_index(['Result_time'], append=True, inplace=True)


# In[ ]:


# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_labs.index.get_level_values('Result_time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For plotting the distribution in days:
plt.figure(figsize=(10, 6))
plt.hist(result_time_hours, bins=30)

plt.xlabel('Result Time (Hours)')
plt.ylabel('Frequency')
plt.title('Distribution of Result Time')
plt.show()

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())


# In[ ]:


# # take the mean of any duplicate index entries for unstacking
# time_series_labs = time_series_labs.groupby(level=[0, 1]).mean()

# # Round up the time-stamps to the next day
# time_series_labs.reset_index(level=1, inplace=True)
# time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')

# # Time shift so that the starting point for each sample is time = 0
# minimum_shifts = time_series_labs.groupby('CSN')['Result_time'].min()
# time_series_labs = time_series_labs.merge(minimum_shifts, left_index=True, right_index=True)
# time_series_labs['Result_time'] = time_series_labs['Result_time_x'] - time_series_labs['Result_time_y']
# time_series_labs.drop(columns=['Result_time_x', 'Result_time_y'], inplace=True)
# time_series_labs.set_index(['Result_time'], append=True, inplace=True)

# # # Calculate timedelta from a reference date (e.g., minimum date)
# # reference_date = time_series_labs['Result_time'].min()  # Or a specific date if needed
# # time_series_labs['Result_time'] = time_series_labs['Result_time'] - reference_date


# In[ ]:


#Proceed with other operations like ceil, set_index, etc.

time_series_labs = time_series_labs.groupby(level=[0, 1]).mean()

# Round up the time-stamps to the next hour
time_series_labs.reset_index(level=1, inplace=True)
time_series_labs.Result_time = time_series_labs.Result_time.dt.ceil(freq='h')
time_series_labs.Result_time = pd.to_timedelta(time_series_labs.Result_time, unit='T')
time_series_labs.set_index('Result_time', append=True, inplace=True)
time_series_labs.reset_index(level=0, inplace=True)

time_series_labs = time_series_labs.groupby('CSN').resample('h', closed='right', label='right').mean().drop(columns='CSN')


# In[ ]:


time_series_labs.reset_index(level=1, inplace=True)
time_series_labs = time_series_labs[time_series_labs['Result_time'].notna()]
time_series_labs.Result_time = pd.to_timedelta(time_series_labs.Result_time, errors='coerce')


# In[ ]:


time_series_labs.update(time_series_labs.groupby(level=0).ffill())
time_series_labs.update(time_series_labs.groupby(level=0).bfill())


# In[ ]:


len(pd.unique(time_series_labs.index.get_level_values(0)))


# In[ ]:


time_series_labs.reset_index(inplace=True)
time_series_labs.set_index(['CSN', 'Result_time'], inplace=True)


# In[ ]:


# Drop missing values samples
missing_samples = time_series_labs[time_series_labs.isnull().any(axis=1)].index.get_level_values(0).tolist()
time_series_labs.drop(missing_samples, level=0, axis=0, inplace=True)


# In[ ]:


# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_labs.index.get_level_values('Result_time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())


# # Vital signs

# In[ ]:


time_series_vitals = pd.read_csv('numerics.csv')


# In[ ]:


# Convert Result_time to datetime, handling out-of-bounds dates
time_series_vitals['Time'] = pd.to_datetime(time_series_vitals['Time'], errors='coerce')

# Remove rows with NaT (Not a Time) in Result_time, which were created due to out-of-bounds dates
time_series_vitals = time_series_vitals[time_series_vitals['Time'].notna()]

# Format Result_time as 'YYYY-MM-DD HH:MM:SS'
time_series_vitals['Time'] = time_series_vitals['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


# Pivot the DataFrame
time_series_vitals = time_series_vitals.pivot_table(
    index=['CSN', 'Time'],  # Index: CSN and Result_time
    columns='Measure',      # Columns: Unique Component_name entries
    values='Value',      # Values: Component_value (or another column like Component_result)
    aggfunc='first'                # Handle duplicates: Take the first occurrence
)


# In[ ]:


# Convert problematic columns to numeric before calculating the mean
for column in time_series_vitals.columns:
    if column != 'CSN' and column != 'Time':  # Exclude non-numeric columns if present
        time_series_vitals[column] = pd.to_numeric(time_series_vitals[column], errors='coerce')


# In[ ]:


time_series_vitals.reset_index(inplace=True)
time_series_vitals['Time'] = pd.to_datetime(time_series_vitals['Time'], errors='coerce')
time_series_vitals.set_index(['CSN', 'Time'], inplace=True)
# Time shift so that the starting point for each sample is time = 0
time_series_vitals.reset_index(level=1, inplace=True)
minimum_shifts = time_series_vitals.groupby('CSN')['Time'].min()
# minimum_shifts = timeseries_summary.time.min(level=0)
time_series_vitals = time_series_vitals.merge(minimum_shifts, left_index=True, right_index=True)
time_series_vitals['Time'] = time_series_vitals['Time_x'] - time_series_vitals['Time_y']
time_series_vitals.drop(columns=['Time_x', 'Time_y'], inplace=True)
time_series_vitals.set_index(['Time'], append=True, inplace=True)


# In[ ]:


# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_vitals.index.get_level_values('Time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For plotting the distribution in days:
plt.figure(figsize=(10, 6))
plt.hist(result_time_hours, bins=30)

plt.xlabel('Result Time (Hours)')
plt.ylabel('Frequency')
plt.title('Distribution of Result Time')
plt.show()

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())


# In[ ]:


time_series_vitals = time_series_vitals.groupby(level=[0, 1]).mean()

# Round up the time-stamps to the next hour
time_series_vitals.reset_index(level=1, inplace=True)
time_series_vitals.Time = time_series_vitals.Time.dt.ceil(freq='h')
time_series_vitals.Time = pd.to_timedelta(time_series_vitals.Time, unit='T')
time_series_vitals.set_index('Time', append=True, inplace=True)
time_series_vitals.reset_index(level=0, inplace=True)

time_series_vitals = time_series_vitals.groupby('CSN').resample('h', closed='right', label='right').mean().drop(columns='CSN')


# In[ ]:


time_series_vitals.reset_index(level=1, inplace=True)
time_series_vitals = time_series_vitals[time_series_vitals['Time'].notna()]
time_series_vitals.Time = pd.to_timedelta(time_series_vitals.Time, errors='coerce')


# In[ ]:


time_series_vitals.update(time_series_vitals.groupby(level=0).ffill())
time_series_vitals.update(time_series_vitals.groupby(level=0).bfill())


# In[ ]:


len(pd.unique(time_series_vitals.index.get_level_values(0)))


# In[ ]:


time_series_vitals.reset_index(inplace=True)
time_series_vitals.set_index(['CSN', 'Time'], inplace=True)


# In[ ]:


# Keep columns with less than 85% missing values
threshold = 0.85
time_series_vitals = time_series_vitals.loc[:, time_series_vitals.isnull().sum() / len(time_series_vitals) < threshold]


# In[ ]:


# Drop missing values samples
missing_samples = time_series_vitals[time_series_vitals.isnull().any(axis=1)].index.get_level_values(0).tolist()
time_series_vitals.drop(missing_samples, level=0, axis=0, inplace=True)


# In[ ]:


# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_vitals.index.get_level_values('Time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())


# In[ ]:


time_series_labs.index = time_series_labs.index.rename('Time', level='Result_time')


# In[ ]:


time_series = time_series_vitals.merge(time_series_labs, left_index=True, right_index=True)


# In[ ]:


visits.drop(['MRN'], axis=1, inplace=True)


# In[ ]:


visits.set_index('CSN', inplace=True)


# In[ ]:


visits = visits[~visits.index.duplicated(keep='first')]


# In[ ]:


final = time_series.merge(visits, left_index=True, right_index=True)


# In[ ]:


# Drop missing values samples
missing_samples = final[final.isnull().any(axis=1)].index.get_level_values(0).tolist()
final.drop(missing_samples, level=0, axis=0, inplace=True)


# In[ ]:


len(pd.unique(final.index.get_level_values(0)))


# In[ ]:


# Check for NaNs
print("Missing values:", final['Age'].isna().sum())

# Check for non-numeric values
non_numeric = final[~final['Age'].astype(str).str.replace('.', '', 1).str.isdigit()]
print("Non-numeric values:\n", non_numeric)

# Convert to numeric, forcing errors to NaN
final['Age'] = pd.to_numeric(final['Age'], errors='coerce')

# Check for negative or unrealistic values
invalid_ages = final[(final['Age'] < 0) | (final['Age'] > 120)]
print("Invalid ages:\n", invalid_ages)


# In[ ]:


# # Group by the first level index ('patient') and get the size of each group
# group_sizes = final.groupby(level=0).size()

# # Filter the group sizes to get patients with more than 24 entries
# patients_to_keep = group_sizes[group_sizes >= 24].index

# # Filter the original DataFrame to keep only the selected patients
# final = final.loc[patients_to_keep]


# In[ ]:


len(pd.unique(final.index.get_level_values(0)))


# In[ ]:


len(final.index.get_level_values(0))


# In[ ]:


final.index.get_level_values(0)


# In[ ]:


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


# In[ ]:


# Extract unique identifiers (CSN) from the original DataFrame
unique_csns = final.index.get_level_values('CSN').unique()

# Split the unique CSNs into train, validation, and test sets
csn_test = np.random.choice(unique_csns, size=int(0.2 * len(unique_csns)), replace=False)
csn_remaining = np.setdiff1d(unique_csns, csn_test)
csn_val = np.random.choice(csn_remaining, size=int(0.25 * len(csn_remaining)), replace=False)
csn_train = np.setdiff1d(csn_remaining, csn_val)

# Subset the original DataFrame based on the split CSNs
df_test = final.loc[csn_test]
df_val = final.loc[csn_val]
df_train = final.loc[csn_train]

# Verify the lengths of the splits
print("Length of df_test:", len(df_test))
print("Length of df_val:", len(df_val))
print("Length of df_train:", len(df_train))

# Extract the indices for each split
test_index = np.where(unique_csns.isin(csn_test))[0]
val_index = np.where(unique_csns.isin(csn_val))[0]
train_index = np.where(unique_csns.isin(csn_train))[0]

# Aggregate or select one row per CSN
df_train_unique = df_train.groupby(level='CSN').first().reset_index()
df_val_unique = df_val.groupby(level='CSN').first().reset_index()
df_test_unique = df_test.groupby(level='CSN').first().reset_index()

# Extract time-to-event and event label for unique CSN
num_durations = 10
labtrans = LabTransform(num_durations)
get_target = lambda df: (df['ED_LOS'].values, df['Outcome'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train_unique))
y_val_surv = labtrans.transform(*get_target(df_val_unique))

# We don't need to transform the test labels
durations_test, events_test = (df_test_unique['ED_LOS'].values, df_test_unique['Outcome'].values)

out_features = labtrans.out_features  # how many discrete time points to predict for (10 here)
cuts = labtrans.cuts

# Verify the lengths of the transformed data
print("Length of y_train_surv:", len(y_train_surv[0]))
print("Length of y_val_surv:", len(y_val_surv[0]))


# In[ ]:


df_train.shape, df_val.shape, df_test.shape, df_train_unique.shape, df_val_unique.shape, df_test_unique.shape


# In[ ]:


final.drop(columns=['ED_LOS', 'Outcome'], inplace=True)


# In[ ]:


for column_index, column_name in enumerate(final.columns):
  print(f"Index: {column_index}, Column Name: {column_name}")


# In[ ]:


def split_sequence_optimized(dataframe, n_steps):
    # Get unique patient IDs
    patient_ids = dataframe.index.get_level_values(0).unique()
    num_patients = len(patient_ids)

    # Pre-allocate the LSTM input array with NaNs
    lstm_input = np.full((num_patients, n_steps, dataframe.shape[1]), np.nan)

    # Iterate through each patient
    for i, patient_id in enumerate(patient_ids):
        # Extract data for the current patient
        patient_data = dataframe.loc[patient_id].values

        # Calculate the starting index for the sequence
        start_index = max(0, patient_data.shape[0] - n_steps)

        # Assign the sequence to the LSTM input array
        lstm_input[i, -patient_data.shape[0]:, :] = patient_data[start_index:]

    # Replace NaNs with the last value in the sequence
    for i in range(lstm_input.shape[0]):  # Iterate over patients
        for j in range(lstm_input.shape[2]):  # Iterate over features
            # Find the last non-NaN value in the sequence
            last_valid_index = np.where(~np.isnan(lstm_input[i, :, j]))[0]
            if len(last_valid_index) > 0:
                last_valid_value = lstm_input[i, last_valid_index[-1], j]
                # Replace NaNs with the last valid value
                lstm_input[i, :, j] = np.where(
                    np.isnan(lstm_input[i, :, j]),
                    last_valid_value,
                    lstm_input[i, :, j]
                )

    return lstm_input


# In[ ]:


# Extract data into timeseries format with 24 timesteps
timeseries_lstm_input = split_sequence_optimized(dataframe=final, n_steps=24)


# In[ ]:


timeseries_lstm_input.shape, np.isnan(timeseries_lstm_input).any()


# In[ ]:


len(test_index), len(train_index), len(val_index)


# In[ ]:


timeseries_lstm_input_train = timeseries_lstm_input[train_index, :, :]
timeseries_lstm_input_val = timeseries_lstm_input[val_index, :, :]
timeseries_lstm_input_test = timeseries_lstm_input[test_index, :, :]

# Quantile transform the features across the population for each timestep
scalers = {}
for i in range(timeseries_lstm_input_train.shape[1]):
    scalers[i] = QuantileTransformer(random_state=0)
    timeseries_lstm_input_train[:, i, :38] = scalers[i].fit_transform(timeseries_lstm_input_train[:, i, :38])

for i in range(timeseries_lstm_input_val.shape[1]):
    timeseries_lstm_input_val[:, i, :38] = scalers[i].transform(timeseries_lstm_input_val[:, i, :38])

for i in range(timeseries_lstm_input_test.shape[1]):
    timeseries_lstm_input_test[:, i, :38] = scalers[i].transform(timeseries_lstm_input_test[:, i, :38])

x_train = timeseries_lstm_input_train.astype('float32')
x_val = timeseries_lstm_input_val.astype('float32')
x_test = timeseries_lstm_input_test.astype('float32')


# In[ ]:


x_train.shape, x_test.shape, durations_test.shape, y_train_surv[0].shape


# In[ ]:


out_features = labtrans.out_features # how many discrete time points to predict for (10 here)
cuts = labtrans.cuts


# In[ ]:


np.save('durations_test_MCMED.npy', durations_test)
np.save('events_test_MCMED.npy', events_test)
np.save('out_features_MCMED.npy', out_features)
np.save('cuts_MCMED.npy', cuts)
np.save('x_train_MCMED.npy', x_train)
np.save('x_val_MCMED.npy', x_val)
np.save('x_test_MCMED.npy', x_test)


# In[ ]:


import pickle

pickle.dump(y_train_surv, open('y_train_surv_MCMED.p', 'wb'))
pickle.dump(y_val_surv, open('y_val_surv_MCMED.p', 'wb'))


# In[ ]:




