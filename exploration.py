### This file is just for EDA purposes

import numpy as np
import pandas as pd
import re

categories_raw = pd.read_csv("C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/data/data/disaster_categories.csv")
messages_raw = pd.read_csv("C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/data/data/disaster_messages.csv")


df = pd.merge(messages_raw, categories_raw, how="left", on="id")
categories = df.categories.str.split(";", expand=True)

df
current_colnames = categories.columns.tolist()
colnames = categories.iloc[0,:].tolist()
clean_colnames = [re.sub("-.*", "", name) for name in colnames]

categories = categories.rename(columns=dict(zip(current_colnames, clean_colnames)))
categories = categories.applymap(lambda x: re.sub(".*-", "", x)).astype("int64")
categories


categories.sum(axis=1).value_counts()


df.drop(columns=["categories"], inplace=True)
df = pd.concat([df, categories], axis=1)
df

### Message
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words("english")

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
  
    return tokens

# Testing the tokenizer on first line
tokenize("The last official estimate several weeks ago said the floods had caused $4.8 billion in damage, had cut the summer grain harvest by 11 million tonnes from last year's and shaved 0.4 percentage point off first-half economic performance.")

### Creating model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()

X_train_counts = vect.fit_transform(df["message"])
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_train_tfidf.toarray().shape
