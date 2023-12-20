import pandas as pd
import spacy
from typing import Set, List, Tuple
from spacy.tokens import DocBin

def df_to_doc_bin(df: pd.DataFrame, categories:list, outfile:str):
    nlp = spacy.blank("en")
    db = DocBin()

    for _, row in df.iterrows():
        doc = nlp.make_doc(row.Question)
        doc.cats = {category: 0 for category in categories}
        doc.cats[row.Category] = 1
        db.add(doc)

    db.to_disk(outfile)

def make_docs(data: List[Tuple[str, str]], target_file: str, cats: Set[str]):
    nlp = spacy.load("en_core_web_md")
    docs = DocBin()

    for doc, label in nlp.pipe(data, as_tuples=True):
        # Encode the labels (assign 1 the subreddit)
        for cat in cats:
            doc.cats[cat] = 1 if cat == label else 0
        docs.add(doc)
    docs.to_disk(target_file)
    return docs

