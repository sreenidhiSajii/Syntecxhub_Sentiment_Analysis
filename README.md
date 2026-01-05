# Sentiment Analysis Project

This is a simple Sentiment Analysis project built using Machine Learning and NLP.  
The model predicts whether a given text expresses a **Positive** or **Negative** sentiment.

The project was developed as part of an AI internship task.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Naive Bayes
- Streamlit

## Project Structure
Syntecxhub_Sentiment_Analysis
├── app.py
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── data/
└── dataset.csv

## How It Works
- Text is cleaned and preprocessed
- Features are extracted using TF-IDF
- A Naive Bayes model is used for classification
- The model predicts sentiment as positive or negative

## How to Run

Install dependencies:
pip install -r requirements.txt

Run CLI version:
python sentiment_analysis.py

Run web app:
streamlit run app.py

## Example
Input:
I really like this product

Output:
Positive