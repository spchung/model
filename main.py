from sklearn.datasets import fetch_20newsgroups

# Define the categories we want to classify
categories = ['sci.space', 'comp.graphics']

# Fetch the training dataset
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# Fetch the testing dataset
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(newsgroups_train.data)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(newsgroups_test.data)


from sklearn.linear_model import LogisticRegression

# Create a logistic regression classifier
clf = LogisticRegression()

# Train the model
clf.fit(X_train_tfidf, newsgroups_train.target)


from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test data
predicted = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(newsgroups_test.target, predicted)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
report = classification_report(newsgroups_test.target, predicted, target_names=newsgroups_test.target_names)
print("Classification Report:\n", report)