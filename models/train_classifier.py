import sys
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix
import pickle
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import MetaData


def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("data/DisasterResponse.db", con=engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    text = re.sub(
        r"[^a-zA-Z0-9]", " ", text.lower()
    )  # normalize case and remove punctuation
    tokens = word_tokenize(text)  # tokenize text
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]  # lemmatize and remove stop words
    return tokens


def build_model():

    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(estimator=RandomForestClassifier())),
        ]
    )

    param_grid = {
        "clf__estimator__n_estimators": [200, 500],
        "clf__estimator__max_features": ["auto", "sqrt", "log2"],
        "clf__estimator__max_depth": [4, 5, 6, 7, 8],
        "clf__estimator__criterion": ["gini", "entropy"],
    }
    cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred))
    print(f"Accuracy: ${np.mean(Y_test.values == y_pred)}")


def save_model(model, model_filepath):
    filehandler = open(model_filepath, "wb")
    pickle.dump(model, filehandler)
    filehandler.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
