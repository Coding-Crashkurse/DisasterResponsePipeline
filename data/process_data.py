import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import os
from sqlalchemy_utils import database_exists, create_database


def load_data(messages_filepath, categories_filepath):
    categories_raw = pd.read_csv(messages_filepath)
    messages_raw = pd.read_csv(categories_filepath)
    df = pd.merge(messages_raw, categories_raw, how="left", on="id")
    return df


def clean_data(df):
    categories = df.categories.str.split(";", expand=True)
    current_colnames = categories.columns.tolist()
    colnames = categories.iloc[0, :].tolist()
    clean_colnames = [re.sub("-.*", "", name) for name in colnames]

    categories = categories.rename(columns=dict(zip(current_colnames, clean_colnames)))
    categories = categories.applymap(lambda x: re.sub(".*-", "", x)).astype("int64")

    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset=["message"], inplace=True)
    df = df[df["related"] != 2]
    return df


def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    if not database_exists(engine.url):
        create_database(engine.url)
        df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
