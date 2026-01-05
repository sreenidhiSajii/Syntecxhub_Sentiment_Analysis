import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.write("Enter text below to predict whether the sentiment is Positive or Negative.")

# Load dataset safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("Unable to load dataset")
    st.exception(e)
    st.stop()

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

user_input = st.text_area("✍️ Enter your text:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == "positive":
            st.success("✅ POSITIVE")
        else:
            st.error("❌ NEGATIVE")

st.markdown("---")
st.caption("Built using Python, Machine Learning & Streamlit")
