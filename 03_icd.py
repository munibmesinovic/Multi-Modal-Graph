from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

pmh = pd.read_csv('pmh.csv')
pmh = pmh[['MRN', 'Code']]

# Filter to MRNs present in visits
mrns_in_visits = visits['MRN'].unique()
pmh = pmh[pmh['MRN'].isin(mrns_in_visits)]

# Get top 500 most frequent codes
top_500_codes = pmh['Code'].value_counts().nlargest(500).index.tolist()
pmh = pmh[pmh['Code'].isin(top_500_codes)]

# Create patient diagnostic sequences for Word2Vec
patient_sequences = pmh.groupby('MRN')['Code'].apply(list).tolist()

# Train Word2Vec on diagnostic sequences
w2v_model = Word2Vec(
    sentences=patient_sequences,
    vector_size=128,
    window=5,
    min_count=10,
    sg=1,  # Skip-gram
    negative=5,
    seed=42,
    workers=4
)

# Create code embeddings matrix
code_embeddings = np.zeros((len(top_500_codes), 128))
code_to_idx = {code: idx for idx, code in enumerate(top_500_codes)}

for code in top_500_codes:
    if code in w2v_model.wv:
        code_embeddings[code_to_idx[code]] = w2v_model.wv[code]

# Compute cosine similarity adjacency matrix
icd_adjacency = cosine_similarity(code_embeddings)

# Save for use in model
np.save('icd_adjacency.npy', icd_adjacency)
np.save('icd_embeddings.npy', code_embeddings)

# Create patient-ICD matrix (multi-hot)
pmh_pivot = pmh.copy()
pmh_pivot['value'] = 1
patient_icd_matrix = pmh_pivot.pivot_table(
    index='MRN', 
    columns='Code', 
    values='value', 
    fill_value=0,
    aggfunc='max'
)

# Ensure all 500 codes are present
for code in top_500_codes:
    if code not in patient_icd_matrix.columns:
        patient_icd_matrix[code] = 0
patient_icd_matrix = patient_icd_matrix[top_500_codes]

visits.set_index('MRN', inplace=True)
visits = visits.merge(patient_icd_matrix, how='left', left_index=True, right_index=True)
visits = visits.fillna(0)
visits.reset_index(inplace=True)
