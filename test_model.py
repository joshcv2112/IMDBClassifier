import numpy as np
import tflearn as tfl
from tflearn.data_utils import pad_sequences
import data_preprocessing as dp

_, testX = dp.convert_reviews()
_, testY = dp.get_sentiment_arrays()

testX = pad_sequences(testX, maxlen=200, value=0.)

# rebuild network structure.
net = tfl.input_data([None, 200])
net = tfl.embedding(net, input_dim=10000, output_dim=128)
net = tfl.lstm(net, 128, dropout=0.8)
net = tfl.fully_connected(net, 2, activation='softmax')
net = tfl.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

model = tfl.DNN(net, tensorboard_verbose=0)
model.load('./saved_models/model1.tfl')

# On my machine, running this prediction shows memory warnings
# These warnings shouldn't actually cause any issues
predictions = model.predict(testX)

results = []
for i in predictions:
    if i[0] > i[1]:
        results.append(0)
    else:
        results.append(1)

count = 0
total_correct = 0
total_incorrect = 0
for i in results:
    if results[count] == testY[count]:
        total_correct += 1
    else:
        total_incorrect += 1
    count += 1

print('\nNetwork accuracy validation against testX dataset: ' + str(total_correct) + "/25000")
