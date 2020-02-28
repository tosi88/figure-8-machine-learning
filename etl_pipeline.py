import pandas as pd
from sqlalchemy import create_engine

# load messages datasets
messages = pd.read_csv("messages.csv")
categories = pd.read_csv("categories.csv")

# merge datasets
df = categories.merge(messages, how='outer', on=['id'])

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
categories.head()

# convert category values to 0 and 1
for column in categories:
    # set each value to be the last character of the string
    categories[column] = pd.Series(categories[column]).astype(str).str.split("-").str[1]

    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column],errors='coerce')
categories.head()

# drop the original categories column from `df`
df = df.drop(["categories"], axis=1)

# concatenate the original dataframe with the new `categories` dataframe
df = df.reset_index(drop=True);
categories = categories.reset_index()
df = pd.concat([df, categories], sort=False, axis=1,ignore_index=False)

# Remove duplicates
print('Dataset contains ' + df.duplicated().sum() + ' duplicate rows')
df.drop_duplicates()
df.duplicated().sum()
print('Duplicates removed')

# insert data to sq-lite database
engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('DisasterResponse', engine, index=False)
print("Merged datasets written to sqlite db: DisasterResponse.db")
