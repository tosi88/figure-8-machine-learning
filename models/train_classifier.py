import sys
from sqlalchemy import create_engine
import pandas as pd


import re
import pickle

import nltk
import nltk
#nltk.download()
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(database_filepath):
    """ load data from sqlite database and split into features (messages) and targets (categories)
    Input:  filepath of the database
    Output: features(X), targets(Y) and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    category_names = df.columns[-36:]
    X = df["message"]
    y = df[category_names]
    return X,y, category_names


def tokenize(text):
    """process text, tokenize, remove stopwords and reduces words to their root form with lemmatizing
    Input: text as string
    Output: cleaned words as array
    """
    # convert to lower case and filter special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words= [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(lemmatizer.lemmatize(word)) for word in words]
    return words

def build_model():
    """
    Build model for classification of messages with help of a sklear pipeline, GridSearch and optimized paramters
    output: model
    """
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(estimator=AdaBoostClassifier()))
    ])

    parameters = {
                'vect__min_df':[1,10,50],
                'clf__estimator__learning_rate': [0.001, 0.01, 0.1],
                'tfidf__smooth_idf': [True, False]
    }

    return GridSearchCV(pipeline, param_grid=parameters, cv=2)

def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate model against test-set with classification report
    Input:
        - model: trained model
        - X_test: test features (
        - Y_test: test targets
        - category_names (categories
    Output: classification report with precision, recall, f1_score and support metrics
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names, digits=2))

def save_model(model, model_filepath):
    """save model to pickle file
    input:
        - model: trained model
        - model_filepath: filepath to save a pickle file
    """
    with open(model_filepath, 'wb') as pickle_file:
                  pickle.dump(model, pickle_file)
    pickle_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
