import pandas as pd

# Load your TSV file
df = pd.read_csv('data/infact_dataset_mapped.tsv', sep='\t')

# Add claim_len and context_len as number of tokens (words)
df['claim_len'] = df['claim_text'].astype(str).apply(lambda x: len(x.split()))
df['context_len'] = df['context'].astype(str).apply(lambda x: len(x.split()))

# Save the updated dataframe
df.to_csv('data/infact_dataset_augmented.tsv', sep='\t', index=False)