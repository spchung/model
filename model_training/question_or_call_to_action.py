import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
import utils
from spacy.cli.train import train as spacy_train

config_path = "question_or_call_to_action_textcat/config.cfg"
output_model_path = "output/question_or_call_to_action"

print("=== Start Training Script ===")
pd.options.display.max_colwidth = None
pd.options.display.max_rows = 6
df = pd.read_csv("question_or_call_to_action_textcat/training_set.csv")
print("=== Data frame Loaded ===")

nlp = spacy.load("en_core_web_md")
print("=== Loaded nlp model ===")

X_train, X_valid, y_train, y_valid = train_test_split(df["Question"].values, df["Category"].values, test_size=0.3)
categories = list(set(df.Category))
print("=== Got Category ===")

utils.make_docs(list(zip(X_train, y_train)), "train.spacy", cats=categories)
utils.make_docs(list(zip(X_valid, y_valid)), "valid.spacy", cats=categories)

print("=== make_docs success ===")
spacy_train(
    config_path,
    output_path=output_model_path,
    overrides={
        "paths.train": "train.spacy",
        "paths.dev": "valid.spacy",
    },
)