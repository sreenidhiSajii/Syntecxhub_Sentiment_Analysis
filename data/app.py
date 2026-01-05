import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.write("Enter text below to predict whether the sentiment is **Positive** or **Negative**.")

# -------------------------------
# Load Dataset (SAFE PATH)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("❌ Unable to load dataset.")
    st.exception(e)
    st.stop()

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------------------
# Train Model
# -------------------------------
X = data["text"]
y = data["sentiment"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("✍️ Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == "positive":
            st.success("✅ Predicted Sentiment: POSITIVE")
        else:
            st.error("❌ Predicted Sentiment: NEGATIVE")

st.m
