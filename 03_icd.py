"""# PAST MEDICAL HISTORY"""

pmh = pd.read_csv('pmh.csv')

pmh = pmh[['MRN', 'Code']]

# Identify MRNs present in the 'visits' DataFrame
mrns_in_visits = visits['MRN'].unique()

# Filter the 'pmh' DataFrame to keep only rows where 'MRN' is in 'mrns_in_visits'
pmh = pmh[pmh['MRN'].isin(mrns_in_visits)]

pmh.set_index('MRN', inplace=True)

# Get the top 500 most frequent codes
top_100_codes = pmh['Code'].value_counts().nlargest(500).index.tolist()

# Filter the pmh DataFrame to keep only rows with codes in the top 500
pmh = pmh[pmh['Code'].isin(top_100_codes)]

# One-hot encode ICD codes
icd_one_hot = pd.get_dummies(pmh['Code'])

# Sum over patients to get patient-ICD matrix
patient_icd_matrix = icd_one_hot.groupby('MRN').sum()

visits.set_index('MRN', inplace=True)

visits = visits.merge(patient_icd_matrix, how='left', left_index=True, right_index=True)
visits = visits.fillna(0)

visits.reset_index(inplace=True)
