# Using the IMDB movie review data set from this link.
# Download, unpack, then place in the project's parent directory.
# http://ai.stanford.edu/~amaas/data/sentiment

# No movie in the dataset has more than 30 reviews,
# since movie reviews tend to follow trends.

# There is not an equal number of movies among
# pos and neg directories, no need to take movies into account.

# imdb.vocab contains all words used in the reviews.

# still accurate
# Max review length for this data set is: 2525

import numpy as np
import pickle
import os
import re
import time


import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb


imdb_vocab_path = './aclImdb/imdb.vocab'
vocab_path = './pickle_files/vocab.pkl'
vocab_ids_path = './pickle_files/vocab_ids.pkl'
sorted_vocab_path = './pickle_files/sorted_vocab.pkl'
vocab_id_dict_path = './pickle_files/vocab_id_dict.pkl'

train_pos_path = './aclImdb/train/pos/'
train_neg_path = './aclImdb/train/neg/'
test_pos_path = './aclImdb/test/pos/'
test_neg_path = './aclImdb/test/neg/'

trainX_path = './pickle_files/trainX.pkl'
trainY_path = './pickle_files/trainY.pkl'
testX_path = './pickle_files/testX.pkl'
testY_path = './pickle_files/testY.pkl'

def get_vocab_array(path):
    lines = []
    try:
        with open(path, 'r') as f:
            lines = f.read().splitlines()
    except Exception:
        print ("IMDB data not found.")
        print ("Download dataset and place in project directory.")
        print ("http://ai.stanford.edu/~amaas/data/sentiment")
    return np.array(lines)

def make_vocab_id_array(vocab_array):
    id = 1
    ids = []
    for term in vocab_array:
        ids.append(id)
        id += 1
    return np.array(ids)

def save_array(path, array):
    with open(path, 'wb') as f:
        pickle.dump(array, f, -1)

def prep_and_save_vocab():
    print('Loading vocab set . . .')
    vocab_array = get_vocab_array(imdb_vocab_path)
    vocab_id_array = make_vocab_id_array(vocab_array)
    save_array(vocab_path, vocab_array)
    save_array(vocab_ids_path, vocab_id_array)
    print('Saved vocab array to ' + vocab_path)
    print('Saved vocab ID array to ' + vocab_ids_path)

# Make this also return an array of pos/neg data
def get_reviews_from_path(path, review_arr, sentiment_arr, sentiment):
    for toot, dirs, files in os.walk(path):
        for filename in files:
            full_path = path + filename
            review = open(full_path, 'r')
            for x in review:
                review_arr.append(x)
                sentiment_arr.append(sentiment)
    return review_arr, sentiment_arr

def prep_and_save_reviews():
    print('Loading movie reviews and sentiment scores . . .')
    train_review_arr = []
    train_sentiment_arr = []
    test_review_arr = []
    test_sentiment_arr = []

    train_review_arr, train_sentiment_arr = get_reviews_from_path(train_pos_path, train_review_arr, train_sentiment_arr, 1)
    train_review_arr, train_sentiment_arr = get_reviews_from_path(train_neg_path, train_review_arr, train_sentiment_arr, 0)
    test_review_arr, test_sentiment_arr = get_reviews_from_path(test_pos_path, test_review_arr, test_sentiment_arr, 1)
    test_review_arr, test_sentiment_arr = get_reviews_from_path(test_neg_path, test_review_arr, test_sentiment_arr, 0)

    save_array(trainX_path, np.array(train_review_arr))
    save_array(trainY_path, np.array(train_sentiment_arr))
    save_array(testX_path, np.array(test_review_arr))
    save_array(testY_path, np.array(test_sentiment_arr))

    print('Movie reviews and sentiment scores loaded.')
    return np.array(train_review_arr), np.array(train_sentiment_arr), np.array(test_review_arr), np.array(test_sentiment_arr)

def get_all_words(list):
    print('Loading all unique words from reviews . . .')
    all_words = []
    for token in list:
        words = token.split()
        for word in words:
            all_words.append(re.sub('[^A-Za-z0-9]+', '', word.lower()))
    print('Loaded all unique words from reviews.')
    return all_words

def count_words(word_arr):
    print('Initializing word count dictionary . . . ')
    d = dict()
    for word in word_arr:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    print('Word count dictionary complete.')
    return d

def sort_word_count(dict):
    print('Sorting word count dictionary . . .')
    x = 0
    sorted_words = []
    while x < 10000:
        max_len = 0
        term = ""
        for word in dict:
            if max_len < dict[word]:
                max_len = dict[word]
                term = word
        sorted_words.append(term)
        del dict[term]
        x += 1
    print('Sorted word count dictionary.')
    return np.array(sorted_words)

def convert_words_to_ints(word_arr):
    new_dict = {}
    i = 0
    for w in word_arr:
        new_dict[w] = i
        i += 1
    return new_dict

def sort_and_save_word_count(dict):
    sorted_arr = sort_word_count(dict)
    new_dict = convert_words_to_ints(sorted_arr)

    save_array(vocab_id_dict_path, new_dict)
    save_array(sorted_vocab_path, sorted_arr)

def integerize_review(review_str, word_id_dict):
    new_arr = []
    x = review_str.split()
    for word in x:
        if word in word_id_dict:
            new_arr.append(word_id_dict[word])
    return new_arr

def integerize_reviews(trainX, word_id_dict):
    new_trainX = []
    for x in trainX:
        int_review = integerize_review(x, word_id_dict)
        new_trainX.append(int_review)
    return new_trainX

def convert_reviews():
    word_id_dict= np.load(vocab_id_dict_path)
    trainX = np.load(trainX_path)
    testX = np.load(testX_path)
    return integerize_reviews(trainX, word_id_dict), integerize_reviews(testX, word_id_dict)

def reformat_sentiment(y):
    new_arr = []
    for i in y:
        new_arr.append(i)
    return new_arr

def get_sentiment_arrays():
    trainY = reformat_sentiment(np.load(trainY_path))
    testY = reformat_sentiment(np.load(testY_path))
    return trainY, testY

def prep_all_data():
    prep_and_save_vocab()
    trainX, trainY, testX, testY = prep_and_save_reviews()
    all_words = get_all_words(testX)
    word_count_dict = count_words(all_words)
    sort_and_save_word_count(word_count_dict)

def call_prep_data():
    t0 = time.clock()
    prep_all_data()
    print ("\nElapsed seconds: " + str(time.clock()))

#call_prep_data()
