import spacy
trained_nlp = spacy.load("output/question_or_call_to_action/model-best")

while True:
    input_text = input("Input:")
    doc = trained_nlp(input_text)
    print(doc.cats)

