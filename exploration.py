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
