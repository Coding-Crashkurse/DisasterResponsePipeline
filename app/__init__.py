from flask import Flask


def tokenize(text):
    """
    Created tokens from raw text
    Args:
    text
    Returns: Lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    text = re.sub(
        r"[^a-zA-Z0-9]", " ", text.lower()
    )  # normalize case and remove punctuation
    tokens = word_tokenize(text)  # tokenize text
    tokens = [
        lemmatizer.lemmatize(word).lower().strip()
        for word in tokens
        if word not in stop_words
    ]  # lemmatize and remove stop words
    return tokens


app = Flask(__name__)
app.secret_key = "markus"

from app import views
