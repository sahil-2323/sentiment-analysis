import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

model= pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer= pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words= set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

def preprocess_text(text):
    text= text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis System")
st.write("Analyze tweet sentiment using Machine Learning & NLP")

twt= st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if twt.strip()== "":
        st.warning("Please enter a tweet")
    else:
        clean_twt= preprocess_text(twt)
        vectorized_twt= vectorizer.transform([clean_twt])
        prediction= model.predict(vectorized_twt)[0]
        
        if prediction=="positive":
            st.success("Positive Sentiment")
        elif prediction=="negative":
            st.error("Negative Sentiment")
        else:
            st.info("Neutral Sentiment")