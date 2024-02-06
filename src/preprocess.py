"""
Module to clean and load data. 
Functions starting with _ are not meant to be called directly but rather are helper functions for use inside the module
"""
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
import nltk
import string  
import langid

nltk.download('punkt')  # Download the necessary resources for tokenization

import os
import pandas as pd

def merge_data_folder(folder_path):
    """
    Load all CSV files in a folder and its subfolders into a Pandas DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing CSV files.

    Returns:
    - df (pd.DataFrame): A DataFrame containing the concatenated data from all CSV files.
    """

    # Initialize an empty DataFrame to store the data
    df = pd.DataFrame()

    # Iterate through the root folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a CSV extension
            if file.endswith(".csv"):
                # Create the full path to the CSV file
                file_path = os.path.join(root, file)
                
                # Load the CSV file into a DataFrame
                csv_data = pd.read_csv(file_path)
                
                # Concatenate the data to the main DataFrame
                df = pd.concat([df, csv_data], ignore_index=True)

    return df

def load_clean_data(path):
    """Loads cleaned data from a CSV file and parses column into their appropriate data types
    """
    df = pd.read_csv(path, sep="|").drop(['Unnamed: 0'],axis=1)
    df["dates"] = pd.to_datetime(df.dates)
    return df

def clean_data(df):
    """
    clean a dataframe of duplicates, non-English tweets, null-valued tweets. 
    Also cleans tweets of html references, links, mentions, hashtags, punctuation, and special characters.
    Args:
        df (pd.DataFrame): dataframe of all tweets

    Returns:
        pd.DataFrame: clean dataframe
    """
    #drop rows with null-value tweets
    df.dropna(subset=["tweets"], inplace=True)
    #drop duplicates
    df.drop_duplicates(subset=["tweet_id"], inplace=True, keep="first")
    # Apply the remove_html_tags function to the specified column
    df['tweets'] = df['tweets'].apply(_remove_html_tags)
    # Apply the remove_links function to the specified column
    df['tweets'] = df['tweets'].apply(_remove_links)
    # Remove paths from tweets
    df['tweets'] = df['tweets'].apply(_remove_paths)
    # Apply the remove_domain_only_links function to the specified column
    df['tweets'] = df['tweets'].apply(_remove_domain_only_links)
    # remove special characters
    df['tweets'] = df['tweets'].map(lambda x: re.sub('[,\.!?]', '', x))
    # remove line breaks
    df['tweets'] = df['tweets'].map(lambda x: re.sub('\n', ' ', x))
    # remove hashtags
    df['tweets'] = df['tweets'].map(lambda x: re.sub(r'#\w+', '', x))
    # remove mentions
    df['tweets'] = df['tweets'].map(lambda x: re.sub(r'@\w+', '', x))
    # remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    df['tweets'] = df['tweets'].map(lambda x: x.translate(translator))
    # remove spaces from leaning column
    df['leaning'] = df['leaning'].map(lambda x: x.replace(" ", ""))
    # make all tweets lowercase
    df['tweets'] = df['tweets'].map(lambda x: x.lower())
    #remove any text between two these brackets <>
    df['tweets'] = df['tweets'].str.replace(r'<.*?>', '', regex=True)
    # Replace empty tweets with NaN
    df['tweets'].replace('', None, inplace=True)
    #again drop rows with null-value tweets
    df.dropna(subset=["tweets"], inplace=True)
    #remove non-english tweets 
    df = df[df["tweets"].apply(_detect_language)]
    # remove tweets that are just spaces
    df = df[~df['tweets'].str.isspace()]
    # Parse the date string
    date_format = "%b %d, %Y · %I:%M %p UTC"
    df["dates"] = pd.to_datetime(df.dates, format=date_format)
    df["n_tokens"] = df["tweets"].apply(_count_tokens)
    return df
    
def cut_into_bins(df, column, bins, labels, new_col_name):
    """adds bins columns to a dataframe 
    """
    df[new_col_name] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df
    
def _detect_language(text):
    lang, confidence = langid.classify(text)
    return lang == 'en'  # Check if the language is English  
 
def _remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()   
 
def _remove_links(text):
    # Define a regular expression to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Replace URLs with an empty string
    return url_pattern.sub('', text)

def _remove_domain_only_links(text):
    # Define a regular expression to match URLs
    url_pattern = re.compile(r'\.(com|net)\b')
    # Replace URLs with an empty string
    return url_pattern.sub('', text)

def _remove_paths(text):
    """
    Remove paths from hyperlinks in the given text.

    Parameters:
    - text (str): Text containing hyperlinks.

    Returns:
    - str: Text with paths removed from hyperlinks.
    """
    # Remove paths from hyperlinks
    cleaned_text = re.sub(r'\S*?/[^ ]*\b', '', text, flags=re.MULTILINE)
    return cleaned_text.strip()

def _count_tokens(sentence):
    """
    Count the number of tokens in a sentence using NLTK word tokenization.

    Parameters:
    - sentence (str): Input sentence.

    Returns:
    - int: Number of tokens in the sentence.
    """
    tokens = nltk.word_tokenize(sentence)
    return len(tokens)

