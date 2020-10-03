# Disaster Response Pipeline Project

## Goals and motivation:

This project wants to make use of a professional, end-to-end data science workflow to get from a raw dataset to a real data science product.
This allows to cover the majority of data science requirements in a single project.
In terms of content, the project deals with how to use the text of a message to assign it to specific content categories. This allows emergency services to react faster and with adequate equipment to a potential disaster.

The project consists of three main parts.

1. ETL Pipeline
2. Machine Learning Pipeline
3. Flask Web Application

### Instructions:

To make this project work on your own machine, run the following commands:
<i>Sidenote: Training the model can take a long time, since we use cross validation to find the best hyperparamters. Please be patient. The final model is not on github, since it´s size exceeds 4GB</i>

1. `git clone https://github.com/Data-Mastery/Udacity-Data-Science-Blog-Post.git` # Creates a copy on your own comput
2. `pip install -r requirements.txt` # Install all dependencies
3. `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` # run an ETL pipeline that cleans data and stores it in a database
4. `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` # Retreive data from database and train a model
5. `cd app` # Go into the app directory
6. `python run.py` # Start Flask App
7. Go to http://localhost:3001/

# Acknowledgements

Thank you for Udacity for providing the main parts of the flask frontend and thank you for [Figure Eight/Appen](https://appen.com/) for providing the dataset.
