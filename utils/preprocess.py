# -*- coding: utf-8 -*-

import nltk
nltk.download('wordnet')
import re
import sklearn
from sklearn.feature_extraction import stop_words
import numpy as np
import pandas as pd


def preprocess_string(string):
    # convert the string to lower case
    string = string.lower()
    
    # extract all tokens in the string (do not include punctuation)
    # Note: the regex permits word-internal apostrophes
    tokens = re.findall(r"\w+(?:[']\w+)*", string, flags=re.UNICODE)
    
    # perform stemming using WordNet lemmatizer
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(token) for token in tokens]
    
    # remove all stop words
    tokens = [token for token in tokens if token not in stop_words.ENGLISH_STOP_WORDS]
    
    # concatenate the tokens into a string, return the string
    new_string = " ".join(tokens)
    return new_string


def preprocess_dataset(dataset):
    # Use this function to preprocess either the headlined or bodies data

    for key in dataset:
            dataset[key] = preprocess_string(dataset[key])

    return dataset


def one_hot(stances):
    # Generate a one-hot encoding for the stances

    # Convert the stances to a numpy array for easy indexing
    array = pd.DataFrame(stances).as_matrix()
    stances_array = np.copy(array[:,2])
    
    # Encode the stances
    stances_array[stances_array == 'agree'] = 0
    stances_array[stances_array == 'disagree'] = 1
    stances_array[stances_array == 'discuss'] = 2
    stances_array[stances_array == 'unrelated'] = 3
    
    one_hot_array = np.zeros((stances_array.size, np.unique(stances_array).size))
    one_hot_array[np.arange(stances_array.size), stances_array.astype(int)] = 1
    
    return one_hot_array