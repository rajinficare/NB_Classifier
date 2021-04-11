import numpy as np
import pandas as pd

VOCAB_SIZE=2500
TOKEN_SPAM_PROB_FILE='resources/prob-spam.txt'
TOKEN_HAM_PROB_FILE='resources/prob-ham.txt'
TOKEN_ALL_PROB_FILE='resources/prob-all.txt'
PROB_SPAM = 0.3116

prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
prob_token_all = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')


def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, freq_idx=3):
    """
    Create a full_matrix from the sparse matrix and return pandas dataframe.

    sparse_matrix --- numpy array
    nr_words --- size of vocabulary
    doc_idx --- position of document id in the sparse matrix. default is 0
    word_idx -- position of word id in the sparse matrix. default is 1
    cat_idx --- postision of category id in the sparse matrix, deafult is 2
    freq_idx -- position of frequency id in the sparse matrix, default is 3
    """
    column_name = ['DOC_ID'] + list(range(nr_words))
    doc_id_names = [0]
    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_name)
    full_matrix = full_matrix.fillna(value=0)

    for i in range(sparse_matrix.shape[0]):  # looping row by row
        # doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        occurence = sparse_matrix[i][freq_idx]

        full_matrix.at[0, 'DOC_ID'] = 0
        # full_matrix.at[0, 'CATEGORY'] = label
        full_matrix.at[0, word_id] = occurence

    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix


def predict(X_features):
    """
    calculates the joint probability compares it and predicts if the email is spam or not spam
    """
    joint_log_spam = X_features.dot(np.log(prob_token_spam) - np.log(prob_token_all)) + np.log(PROB_SPAM)
    joint_log_ham = X_features.dot(np.log(prob_token_ham) - np.log(prob_token_all)) + np.log(1 - PROB_SPAM)
    y_predict = (joint_log_spam > joint_log_ham)*1
    # converting boolean to integer
    if y_predict == 0:
        return 0, "THIS IS NOT A SPAM EMAIL"
    else:
        return 1, "ALERT!!YOU HAVE A SPAM EMAIL!!"