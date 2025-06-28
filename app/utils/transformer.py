import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        clean_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens if word.isalnum() and word not in self.stop_words
        ]
        return " ".join(clean_tokens)

    def preprocess_text(self) -> pd.DataFrame:
        self.df["processed_message"] = self.df["message"].apply(self.tokenize_and_lemmatize)
        return self.df

    def encode_labels(self) -> pd.DataFrame:
        le_agent = LabelEncoder()
        le_sentiment = LabelEncoder()
        self.df["agent_encoded"] = le_agent.fit_transform(self.df["agent"])
        self.df["sentiment_encoded"] = le_sentiment.fit_transform(self.df["sentiment"])
        return self.df

    def transform_all(self) -> pd.DataFrame:
        self.df = self.preprocess_text()
        self.df = self.encode_labels()
        return self.df
