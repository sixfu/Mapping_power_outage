"""
Here we are bringing in our saved models from 04_models.py and 04b_lstm_rnn.ipynb to classify
streammed tweets.  Predictions made will on each of the previously saved models which will be
fed into our multi-layed model 'labels' for a prediction of 1 if all other models predict 1.
"""

import pandas as pd
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

preprocess = __import__('02_preprocess') # Bring in the 02_preprocess.py file
neural = __import__('05b_neural') # Bring in the 05b_neural.py file

np.random.seed(42)

# This will run the twitter_bot.py script
exec(open("05_twitter_bot.py").read())

# Bring in out csv from twitter_bot.py
out = pd.read_csv('../data/out.csv')

# out.to_csv('../data/baseline_checker.csv')
print(out.shape)

# Run through preprocessing
out['tweet'] = out['tweet'].apply(preprocess.clean_text)
X = out['tweet']

# Load in the LogisticRegression
model_lr = pickle.load(open("gridsearch_lr_model.sav", "rb"))

# Load in the Naive Bayes Model
model_nb = pickle.load(open("gridsearch_nb_model.sav", "rb"))

# Load in the Random Forest Model
model_rf = pickle.load(open("gridsearch_rf_model.sav", "rb"))

# Predicting
pred_lr = model_lr.predict(X)
pred_nb = model_nb.predict(X)
pred_rf = model_rf.predict(X)
pred_rnn = neural.neural(out)

# Adding prediction columns
out['predict_lr'] = pred_lr
out['predict_nb'] = pred_nb
out['predict_rnn'] = pred_rnn
out['predict_rf'] = pred_rf

# Loop through each prediction and make a new column with final model
# prediction if all models are predicting a power outage
labels = []
for index, row in out.iterrows():
    if row['predict_lr'] + row['predict_nb'] + row['predict_rnn'] + row['predict_rf'] > 3.5:
        # the random forest model outputs a float, which is why our condition is not 4.
        labels.append(1)
    else:
        labels.append(0)
out['labels'] = labels

# Seeing the spread of 0's and 1's
print(out['predict_lr'].value_counts(normalize=True))
print(out['predict_nb'].value_counts(normalize=True))
print(out['predict_rf'].value_counts(normalize=True))
print(out['predict_rnn'].value_counts(normalize=True))
# Saving the final csv
out.to_csv('../data/predictions.csv', index=False)
