import pandas as pd
from sentence_transformers import SentenceTransformer

# get data
df = pd.read_csv('http://bit.ly/dataset-sst2',
                 nrows=100, sep='\t', names=['text', 'label'])

# data cleaning
df['label'] = df['label'].replace({0: 'negative', 1: 'positive'})

# start embedding
sentence_bert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def get_embeddings(sentences):
    return sentence_bert_model.encode(sentences, batch_size=32, show_progress_bar=True)

embedding = get_embeddings(df['text'])

# embedding to Dataframe
embedding_df = pd.DataFrame(embedding)
embedding_df.to_csv('output.tsv', sep='\t', index=None, header=None)
df.to_csv('metadata.tsv', index=False, sep='\t')