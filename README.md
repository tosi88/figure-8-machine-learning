# Disaster Response Pipeline Project
This should provide a solution for the Udacity data science nano-degree project. The project consists of messaging data for disasters from Figure-eight.

## 1. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

## 2. Project Motivation
The underlying data set is a set of emergency text messages. In case of major catastrophic events, incidents are reported by thousands of people. In order to distinguish already known incidents and information from newly incoming relevant news, we use a machine learning pipeline.
First we clean the data set in an ETL pipeline and afterwards we use the pipeline to train a model that is able to decide weather newly incoming messages provide relevant information.

## 3. Structure
The repo contains the following folders:
- data/: contains the raw data, i.e. message.csv and category.csv as well as a data processing script that processes the data from those csv files and writes them to an sqlite database.
- model/: contains the python script for training a machine learning model and an already trained model.
- app/: contains the backend script.

## 4. Licensing, Authors, Acknowledgements, etc.
MIT Licensing - special thanks to Udacity and Figure-Eight for providing the data set.
