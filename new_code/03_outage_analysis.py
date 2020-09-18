"""
After preprocessing the data we will now classify our tweets. 0 if the tweet is not about
power outages and 1 if the tweet is about power outages. If a tweet contains a
word in the bad_words list it will be automatically classified as a 0.
"""
import pandas as pd
import numpy as np

preprocess = __import__('02_preprocess') # Bring in the 02_preprocess.py file

np.random.seed(42)

# Words we do not want to classify as a power outage.
# Some of these are from a TV show "Power" that we do not want to include in our list.
bad_words = ["Omari","Hardwick","50","cent","Joseph","Rotimi","Naturi","Naughton","Lela","Loren",
"Larenz","Tate","internet","Internet","Shane","Johnson","Jerry","Ferrara","Lucy","Walters",
"Sinqua","Walls","Andy","Bean","David","Fumero","Sung","Kang","kidding","jk","Ghost", "politics",
"trump", "biden" "wifi's", "attorney", "point", "powerpoint", "girl", "guy", "black", "white", "austin",
"abusing", "abused", "Starz", "starz",  "ruiz", "estelle", "kanan", "dre", "wifi", "hotel", "woman",
"man", "guess", "potency", "competenecy", "competence", "brawn", "horsepower", "hp", "acceleration",
"grunt", "vacation", "resort", "GOP", "nazi"]

# Read in the done_processing.csv file from preprocess.py
final_csv = pd.read_csv('../data/done_processing.csv')
tweets = []

for xi in final_csv['tweet']:
    tweets.append(xi)
label = []

# Adding a 0 or 1 based on the bad_words list
for test in tweets:
    res = any(ele in test for ele in bad_words)
    if res == True: # Will check if a word in bad_words is in a tweet if so = 0
        label.append(0)
    else:
        label.append(1)

final_csv['label'] = label

# Read in bad words csv's here!
bad_words1 = pd.read_csv('../data/bad_words1.csv')
bad_words2 = pd.read_csv('../data/bad_words2.csv')

# Label all of these as non power outage tweets
bad_words1['label'] = 0
bad_words2['label'] = 0

# Drop any NaN values
bad_words1.dropna(inplace=True)
bad_words2.dropna(inplace=True)

# Applting the clean_text function from preprocess to the pulled bad tweets
bad_words1['tweet'] = bad_words1['tweet'].apply(preprocess.clean_text)
bad_words2['tweet'] = bad_words2['tweet'].apply(preprocess.clean_text)
final_csv = final_csv.append([bad_words1, bad_words2])
print(final_csv.head())

# print(label)
print(final_csv.shape)
print(final_csv['label'].value_counts())
print(final_csv['label'].value_counts(normalize=True))

# Export data to be used in 04_models.py and 04b_lstm_rnn.ipynb
final_csv.to_csv('../data/ready_for_modeling.csv', index=False)
