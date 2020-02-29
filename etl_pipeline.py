import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
   """ load and merge message and categories file.
   Input:
   messages.csv and categories.csv filepath
   Output:
   dataframe containing the two data types joined by message id.
   """
   messages = pd.read_csv(messages_filepath, delimiter=",", encoding = 'utf-8')
   categories = pd.read_csv(categories_filepath,  delimiter=",", encoding = 'utf-8')
   df = messages.merge(categories, how='inner', on= 'id')
   return df

def clean_data(df):
    """ cleans the merged dataframe.
    Input:
    the data frame loaded with load_data function
    Output:
    dataframe with messages, categories as column names and without duplicates
    """

    print('Cleaning merged data frame')
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)

    # extract headers
    headings = categories.loc[0].str.split("-");
    header = {}
    i=0
    for heading in headings:
        header[i] = heading[0]
        i=i+1

    # rename headers
    categories = categories.reset_index().rename(columns = header)
    categories = categories.drop(['index'], axis=1)

    # convert category values to 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.Series(categories[column]).astype(str).str.split("-").str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column],errors='coerce')

    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.reset_index(drop=True);
    categories = categories.reset_index()
    df = pd.concat([df, categories], sort=False, axis=1,ignore_index=False)

    # Remove duplicates
    df.drop_duplicates()

    return df

def save_data(df):

    """ save the cleaned dataframe to an sqlite database.
    Input:
    df: cleaned dataframe
    Output:
    saved database in the data folder
    """
    try:
        print('Saving data to sqlite database')
        engine = create_engine('sqlite:///DisasterResponse.db')
        df.to_sql('DisasterResponse', engine, index=False)
        print("Merged datasets written to sqlite db: DisasterResponse.db")
    except ValueError:
        print("Data already written to database")

def main():
    if len(sys.argv) == 3:

        messages_filepath, categories_filepath = sys.argv[1:]

        print('Loading data from files\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        df = clean_data(df)

        save_data(df)

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively. '\
              '\n\nExample: python etl_pipeline.py '\
              'disaster_messages.csv disaster_categories.csv ')


if __name__ == '__main__':
    main()
