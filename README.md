## Intelligent Systems Final Project
## Joshua Vaughan
## A01490610

## IMDB Movie Review Sentiment Analysis

For this project I used Deep Neural Networks to classify movie reviews from the IMDB dataset to determine their sentiment. Movies were ranked and labeled as either 1 (positive sentiment) or 0 (negative sentiment).

## STEP 1

The processed IMDB data is stored in pickle files, but due to the large size of the dataset, they cannot be store in this repository.
Download the pickle files from this link:
  https://drive.google.com/drive/u/2/folders/16HeENgcR249G_egWNpySWAjhdYj95ezj

## STEP 2 

Place the folder of pickle files in the project directory, make sure it is named "pickle_files", or else the program won't be able to find them.

## STEP 3

Running this project requires either Python 2 or 3. I have run the project on both Ubuntu and Windows 10, it should work for whatever OS you are using.

The following packages also need to be installed:
  tflearn
  tflearn.datautils
  numpy
  pickle

## STEP 4

The directory 'saved_models' contains the already trained Deep Neural Network on the training datasets. To test this trained model, just run "python test_model.py". This may take a moment to execute as it tests the DNN against 25,000 distinct movie reviews and prints the accuracy of the model. It shouldn't take more than a minute or two to finish execution.

## STEP 5 (optional)

If you wish to re-train the network, you first must download the original data source. The IMDB data source can be dowloaded from:
  http://ai.stanford.edu/~amaas/data/sentiment
Once at the link above, click on the hyper link that says "Large Movie Review Dataset v1.0". This will download a set of 50,000 movie reviews from IMDb, half for training, and half for validation. Once downloaded, unzip the package and place it in the main project directory, also be sure this extracted folder is named "aclImdb".

## STEP 6 (optional)

Now to process all the raw movie review data so it can be fed through the DNN. 
Start the python interpreter, and run the following commands:
  >> import data_preprocessing as dp
  >> dp.call_prep_data()
This may take several minutes, depending on your system hardware.

## STEP 7 (optional)

To verify the validity of all files, run unit_tests.py to see that all data is processed and available in the pickle files for the network to train and validate on.

## STEP 8 (optional)

Finally, run train.py to train the network. It will automatically be saved in the saved_models directory upon completion.

## MY RESULTS

When I ran this network on the train data, it achieved a whopping 98% accuracy after 10 epochs. When validating on the unique validation datasets, it reached a 70% accuracy. The lesser accuracy on the validation set is due to the fact that the network didn't train on this different dataset which has entirely distinct reviews.
