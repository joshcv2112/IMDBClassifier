# This script tells you the longest reviews in the data set.

import os
from nltk.tokenize import RegexpTokenizer

train_pos_path = './aclImdb/train/pos/'
train_neg_path = './aclImdb/train/neg/'
test_pos_path = './aclImdb/test/pos/'
test_neg_path = './aclImdb/test/neg/'

# Actually, should probably tokenize the data before counting words
# put this out to a function and run it for each review directory
# then count the lengths of each review, returning the max length

# Use this link to help strip punctuation from reviews when tokenizing.
# https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer

def tokenize_review(path):
    review = open(path, 'r')
    review_txt = review.read()
    tokenizer = RegexpTokenizer(r'\w+')
    review_tokens = tokenizer.tokenize(review_txt)
    return review_tokens

def print_all_reviews():
    for root, dirs, files in os.walk(train_pos_path):
        for filename in files:
            full_path = train_pos_path + filename
            review = open(full_path, 'r')
            for x in review:
                print(x)

def get_max_length(path):
    max = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            full_path = path + filename
            tokenized_review = tokenize_review(full_path)
            if len(tokenized_review) > max:
                max = len(tokenized_review)
    return max

def print_review_lengths():
    print('START')
    print ('Max review length: ' + str(max(
        get_max_length(train_pos_path),
        get_max_length(train_neg_path),
        get_max_length(test_pos_path),
        get_max_length(test_neg_path)
    )))
    print('END')


print_review_lengths()
print_all_reviews()
