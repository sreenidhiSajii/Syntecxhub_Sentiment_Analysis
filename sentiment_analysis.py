print("### FINAL VERSION RUNNING ### - sentiment_analysis.py:1")

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/dataset.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["sentiment"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

predictions = model.predict(X_vec)
accuracy = accuracy_score(y, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}% - sentiment_analysis.py:31")

while True:
    user_input = input("\nEnter text (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)

    print("Predicted Sentiment: - sentiment_analysis.py:43", result[0])
