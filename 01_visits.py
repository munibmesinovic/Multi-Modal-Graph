"""# VISITS"""

visits = pd.read_csv('visits.csv')

print(f"Number of unique patients: {visits['MRN'].nunique()}")

print(f"Number of unique visits: {visits['CSN'].nunique()}")

visits.isnull().sum() / len(visits)

# Remove those with missing race, ethnicity, triage measurement, length of stay in ED

visits = visits.dropna(subset=['Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2', 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS'])
visits = visits[['MRN', 'CSN', 'Age', 'Gender', 'Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2',
                 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo']]

print(f"Number of unique patients: {visits['MRN'].nunique()}")

print(f"Number of unique visits: {visits['CSN'].nunique()}")

one_hot = pd.get_dummies(visits['Race'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Race',axis = 1)
# Join the encoded df
visits = visits.join(one_hot)

one_hot = pd.get_dummies(visits['Gender'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Gender',axis = 1)
# Join the encoded df
visits = visits.join(one_hot)

one_hot = pd.get_dummies(visits['Ethnicity'], dtype=int)
# Drop column B as it is now encoded
visits = visits.drop('Ethnicity',axis = 1)
# Join the encoded df
visits = visits.join(one_hot, rsuffix='_ethnicity')

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

visits = visits[visits['ED_LOS'] <= 24]
