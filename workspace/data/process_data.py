import sys
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT: messages_filepath & categories_filepath
    OUTPUT: a merged df which is the conbination of messages & categories files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left',on=['id'])
    return df

    


def clean_data(df):
    '''
    INPUT: a dataframe
    OUTPUT:
    - split the 'categories' column into respectives categories splited by ';'
    - get the values of each category
    - change the data type into 'int'
    - drop the old 'categories' column and add the new columns into the df.
    Result: a cleaned df
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    categories.columns = [x[:-2] for x in categories.iloc[0,:]]
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda x: x[len(x)-1:])
        categories[col] = categories[col].astype(int)
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
#     df = df.iloc[:500,:] #To test
    return df



def save_data(df, database_filename):
    '''
    INPUT: a df and the database_filename which contains the df later.
    OUPUT: the df is saved in the database
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    return df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)  


def main():
    '''
    INPUT: None
    OUTPUT: Perform 3 steps of Data processing: Load, Clean & Save data.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('messages_filepath: {} \n, categories_filepath: {} \n, database_filepath: {} \n'.format(messages_filepath, categories_filepath, database_filepath))

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print(len(sys.argv))
        print(sys.argv)
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
