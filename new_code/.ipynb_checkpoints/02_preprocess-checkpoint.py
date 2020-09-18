"""
This file needs to preprocess the words in each tweet that we brought in from 01_scraping_tweets.
After bringing in the file we will clean all of the text and get rid of the unnecessary characters and
numbers. We will then save the clean version of the csv to a new file called done_processing.csv that will
move onto 03_outage_analysis for future steps.
"""
import pandas as pd
import numpy as np
import re

np.random.seed(42)

# Bringing if the data
csv = pd.read_csv('../data/ready_for_processing.csv')

# These will be replaced by a space ' '
symbol_replace_space = re.compile('[/(){}\[\]\|@,;.£]')

# We will get rid of all these in the function below
bad_symbols = re.compile('[^0-9a-z #+_]')

def clean_text(tweet):

    # Make all of the text lower case
    tweet = tweet.lower()

    # Replace symbol_replace_space with a space
    tweet = symbol_replace_space.sub(' ', tweet)

    # Replace bad_symbols with a space
    tweet = bad_symbols.sub('', tweet)

    # This gets rid of the integers
    tweet = re.sub(r'\d+', '', tweet)

    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet

# Applying the clean_text function above to every tweet
csv['tweet'] = csv['tweet'].apply(clean_text)

# Export data for 03_outage_analysis.py
csv.to_csv('../data/done_processing.csv', index=False)
