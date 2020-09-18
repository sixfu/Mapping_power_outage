"""
This script enables us to run streamed Tweets to predict an outage using our LSTM RNN model.
New tweets will be preprossed removing unnecessary characters and numbers, and tokenized that will
feed into our model.

Our model will then make a prediction on each tweet in 06_classify_tweets.py
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
np.random.seed(42)

def neural(out):

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 280 # First 280 words in each title
    # This is fixed.
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)  # instantiating Tokenizer
    tokenizer.fit_on_texts(out['tweet'].values)  #fit text to values
    word_index = tokenizer.word_index  # replaces every word
    X = tokenizer.texts_to_sequences(out['tweet'].values) # adds index # of every word in a title
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)   # pads each post so every title has 280 length
    model = tf.keras.models.load_model('./saved_model/lstm_rnn_model_')

    # make a prediction
    ls = []
    for ea_tweet in out['tweet']:
        new_post = [ea_tweet]
        seq = tokenizer.texts_to_sequences(new_post)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        labels = [0, 1]
        ls.append(labels[np.argmax(pred)])


    return ls
