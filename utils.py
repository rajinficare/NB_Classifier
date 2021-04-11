# notebook imports

from os import walk
from os.path import join

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup


## Re defining the Function to clean and preprocess the email and also remove html tags (putting it altogether)

def clean_message(msg, stemmer=PorterStemmer(), stop_words=set(stopwords.words('english'))):
    filtered_words = []

    soup = BeautifulSoup(msg, 'html.parser')
    msg_no_html = soup.get_text()

    # convert to lower case and splits the individual words

    words = word_tokenize(msg_no_html.lower())

    for word in words:
        # remove stopwords and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

    if filtered_words:
        return filtered_words
    else:
        return "Please write the email having stop_words"



def make_dataframe(word_list):
    word_columns_df = pd.DataFrame.from_records(word_list)
    return word_columns_df


def make_sparse_matrix(df, indexed_words, labels=0):
    """
    returns sparse matrix as a dataframe.
    df : Dataframe with words as columns and document_id as index (X_train or X_test)
    indexed_words : index of words ordered by word_id
    labels: category as a series (y_train or y_test)

    """
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for i in range(nr_rows):
        for j in range(nr_cols):
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels

                item = {'LABEL': category, 'DOC_ID': doc_id, 'OCCURENCES': 1, 'WORD_ID': word_id}
                dict_list.append(item)

    sparse_df = pd.DataFrame(dict_list)
    return sparse_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum().reset_index()

