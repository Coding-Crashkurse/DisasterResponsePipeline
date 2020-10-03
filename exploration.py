### This file is just for EDA purposes

import numpy as np
import pandas as pd
import re

categories_raw = pd.read_csv("C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/data/disaster_categories.csv")
messages_raw = pd.read_csv("C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/data/disaster_messages.csv")


df = pd.merge(messages_raw, categories_raw, how="left", on="id")
categories = df.categories.str.split(";", expand=True)

df
current_colnames = categories.columns.tolist()
colnames = categories.iloc[0,:].tolist()
clean_colnames = [re.sub("-.*", "", name) for name in colnames]

categories = categories.rename(columns=dict(zip(current_colnames, clean_colnames)))
categories = categories.applymap(lambda x: re.sub(".*-", "", x)).astype("int64")

df.drop(columns=["categories"], inplace=True)
df = pd.concat([df, categories], axis=1)
df.drop_duplicates(subset=["message"], inplace=True)
df = df[df["related"] != 2]


df["related"].value_counts()

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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
clf = RandomForestClassifier()

X = df["message"]
y = df.iloc[:, 4:]

X_train, X_test, y_train, y_test = train_test_split(X, y)

## Test multilabel CM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix


pipeline = Pipeline([
('vect', CountVectorizer(tokenizer=tokenize)),
('tfidf', TfidfTransformer()),
('clf', MultiOutputClassifier(estimator=RandomForestClassifier())),
])

param_grid = { 
    'clf__estimator__n_estimators': [100, 200, 500]
}

cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)

cv.fit(X_train, y_train)
X_test
  
y_pred = cv.predict([X_test.iloc[0]])
X_test.iloc[0]
y_pred
X_test.iloc[0]


import pickle

filehandler = open("C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline", "wb")

pickle.dump(cv, "C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/classifier.pkl")

classification_report(y_test.values, y_pred)

np.mean(y_test.values == y_pred)
mcm = multilabel_confusion_matrix(y_test, y_pred)

accuracy_per_col = [(matrix[1, 1] + matrix[0, 0]) / sum(map(sum, matrix)) * 100 for matrix in mcm]
accuracy_per_col

df.head()
genre_counts = df.groupby("genre").count()["message"]
genre_names = list(genre_counts.index)

####

categories = df.drop(columns=["id", "message", "original", "genre"])
stacked_categories = (
    categories.stack()
    .reset_index()
    .rename(columns={0: "count", "level_1": "categories"})
)

categories_count = (
    stacked_categories.drop(columns=["level_0"])
    .groupby("categories")
    .sum()
    .sort_values(by="count", ascending=False)
)
categories_list = list(categories_count.index)


genre_counts = df.groupby("genre").count()["message"]
genre_names = list(genre_counts.index)


categories_count["count"]
genre_counts

import joblib

joblib.dump(cv, 'C:/Users/User/Desktop/Udacity Nanodegree/Projects/DisasterResponsePipeline/models/classifier.pkl') 
