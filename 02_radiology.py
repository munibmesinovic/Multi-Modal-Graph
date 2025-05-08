"""# X-Ray Reports"""

rads = pd.read_csv('rads.csv')

rads.set_index('CSN', inplace=True)

# Drop rows with missing values in 'Study' or 'Impression'
rads = rads.dropna(subset=['Study', 'Impression'])

rads['Text'] = rads['Study'] + ' ' + rads['Impression']

rads.drop(['Study', 'Impression', 'Order_time', 'Result_time'], axis=1, inplace=True)

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

rads_with_embeddings2 = rads_with_embeddings.copy()

rads_with_embeddings2.drop(['Text'], axis=1, inplace=True)

rads_with_embeddings2

visits.set_index('CSN', inplace=True)
rads_with_embeddings2.set_index('CSN', inplace=True)

visits = visits.merge(rads_with_embeddings2, how='left', left_index=True, right_index=True)
visits = visits.fillna(0)

visits.reset_index(inplace=True)