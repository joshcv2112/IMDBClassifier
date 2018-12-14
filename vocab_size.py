# This file prints the length of the vocabulary set for this dataset.

train_pos_path = './aclImdb/train/pos/'
word_file = open('./aclImdb/imdb.vocab', 'r')
word_arr = []

for x in word_file:
    word_arr.append(x)
    if x == 's':
        print('GOT ONE')

print ('The length of this data set\' vocabulary is ' + str(len(word_arr)))
