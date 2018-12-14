import pickle
import numpy as np
import os, re, time
import tflearn as tfl
import data_preprocessing as dp
from tflearn.data_utils import to_categorical, pad_sequences


def train_and_save_model():
    # Run this if pkl files already exist in directory pickle_files
    trainX, testX = dp.convert_reviews()
    trainY, testY = dp.get_sentiment_arrays()

    # AVG REVIEW LENGTH: 165.3178

    # REMOVE THIS JUNK
    print('trainX ' + str(trainX[0]))
    print('trainX ' + str(len(trainX[0])))
    print('trainY ' + str(trainY[0]))
    print('trainY ' + str(type(trainY[0])))

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=200, value=0.)
    testX = pad_sequences(testX, maxlen=200, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    deep_net = tfl.input_data([None, 200])
    deep_net = tfl.embedding(deep_net, input_dim=10000, output_dim=128)
    deep_net = tfl.lstm(deep_net, 128, dropout=0.8)
    deep_net = tfl.fully_connected(deep_net, 2, activation='softmax')
    deep_net = tfl.regression(deep_net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training 1ST RUN
    model = tfl.DNN(deep_net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32, n_epoch=20)

    model.save('./saved_models/model1.tfl')



t0 = time.clock()

train_and_save_model()



print ("\nElapsed seconds: " + str(time.clock()))
