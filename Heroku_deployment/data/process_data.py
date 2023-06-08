import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Dataframe from messages and categories filepaths.
    INPUT
        messages_filepath (string): Filepath to messages.csv file.
        categories_filepath (string): Filepath to categories.csv file.

    OUTPUT
        df (DataFrame) : Merged DataFrame (Merge the messages and categories dataframes using the common id).
    """

    # 1. Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # 2. Load categories dataset
    categories = pd.read_csv(categories_filepath)
    # 3. Merge the messages and categories datasets using the common id
    df = messages.merge(categories, on="id")

    return df


def clean_data(df):
    """
    Clean the DataFrame as follows:
        1.  Split categories into separate category columns.
        2.  Convert category values from string to numeric (just numbers 0 or 1).
        3.  Remove duplicates.
    INPUT
        df (DataFrame): The dataFrame created by `load_data function`.

    OUTPUT
        df (DataFrame): the cleaned dataset.
    """

    # 1. Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # 2. Select the first row of the categories dataframe
    row = categories.loc[0]

    # 3. Use this row to extract a list of new column names for categories.
    # Apply a lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda i: i[:-2])

    # 4. Rename the columns of `categories`
    categories.columns = category_colnames
    
    # 5. Convert category values to just numbers 0 or 1.
    for column in categories:
        # 5.1 Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # 5.2 Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # 6. Drop the original categories column from `df`
    df = df.drop(columns="categories")

    # 7. Concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)

    # 8. Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an SQLite database using pandas `to_sql` method combined with the SQLAlchemy library.
    Args:
        df (DataFrame): the cleaned dataframe (cleaned by clean_data function).
        table_name (str): name of the SQLite table.
        database_filename (str): Name of the SQLite database file.
    """

    engine = create_engine("sqlite:///" + database_filename)

    # Create table `Disaster`. Replace it if exists (default='fail')
    df.to_sql("Disaster", engine, if_exists='replace', index=False)  

    df.to_csv("data/disaster_clean.csv",index=False)


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
