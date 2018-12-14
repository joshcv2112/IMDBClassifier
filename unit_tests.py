import os
import unittest
import data_preprocessing as dp

# All data paths for movie reviews
imdb_vocab_path = './aclImdb/imdb.vocab'
vocab_path = './pickle_files/vocab.pkl'
vocab_ids_path = './pickle_files/vocab_ids.pkl'
sorted_vocab_path = './pickle_files/sorted_vocab.pkl'
vocab_id_dict_path = './pickle_files/vocab_id_dict.pkl'
trainX_path = './pickle_files/trainX.pkl'
trainY_path = './pickle_files/trainY.pkl'
testX_path = './pickle_files/testX.pkl'
testY_path = './pickle_files/testY.pkl'

class LearningCase(unittest.TestCase):
    # Tests that all testing/validation data is as it should be.
    def test_convert_reviews(self):
        trainX, testX = dp.convert_reviews()
        trainY, testY = dp.get_sentiment_arrays()
        assert len(trainX) == 25000, "trainX data not proper size or format"
        assert len(testX) == 25000, "testX data not proper size or format"
        assert len(trainY) == 25000, "trainY data not proper size or format"
        assert len(testY) == 25000, "testY data not proper size or format"

    # test verifies that all data directories are where they should be and named properly.
    def test_data_paths(self):
        assert os.path.isfile(imdb_vocab_path), ("File: " + imdb_vocab_path + " not found...")
        assert os.path.isfile(vocab_path), ("File: " + vocab_path + " not found...")
        assert os.path.isfile(vocab_ids_path), ("File: " + vocab_ids_path + " not found...")
        assert os.path.isfile(sorted_vocab_path), ("File: " + sorted_vocab_path + " not found...")
        assert os.path.isfile(vocab_id_dict_path), ("File: " + vocab_id_dict_path + " not found...")
        assert os.path.isfile(trainX_path), ("File: " + trainX_path + " not found...")
        assert os.path.isfile(trainY_path), ("File: " + trainY_path + " not found...")
        assert os.path.isfile(testX_path), ("File: " + testX_path + " not found...")
        assert os.path.isfile(testY_path), ("File: " + testY_path + " not found...")

def main():
    unittest.main()

if __name__ == "__main__":
    main()
