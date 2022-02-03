import sys
# import libraries
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.metrics import confusion_matrix
# from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import pickle



def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
#     category_names = Y.columns
    return X,Y


def tokenize(text):
    '''
    INPUT: a sentence.
    OUTPUT: an array of words after going through the text processing 
    (Normalize, Tokenize, Remove stopwords, stemmed & lemmatize)
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = text.split()
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__binary': [False,True],
        'vect__ngram_range': [(1, 1),(1,2),(2,3)],
#         'clf__estimator__n_estimators': [10,11,12]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, y_test, y_pred):
    print('1. CLASSIFICATION REPORT:')
    i=0
    for col in y_test:
        print('Feature {}: {} \n'. format(i+1, col))
        print(classification_report(y_test[col], y_pred[:,i]))
        i+=1
    accuracy = (y_pred == y_test).mean()
    print("2. ACCURACY: ")
    print(accuracy)
    print("3. Best Parameters:", model.best_params_)


def save_model(model, model_filepath):
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('database_filepath: {}\n model_filepath: {}\n\n'.format(database_filepath, model_filepath))
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
#         evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, Y_test, model.predict(X_test))

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