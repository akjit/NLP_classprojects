#!/usr/bin/env python
# import the required packages here
import numpy as np 
import sys
import math
import time


def readData(filename):
    datafile = open(filename, 'r')
    data = np.genfromtxt(filename,delimiter = ",")
    datafile.close()
    return data

def count(data):
    onion = 0
    for i in data:
        if i == 1:
            onion = onion + 1
    return onion

def trainBayes(xtrain, ytrain):
    doc_size = ytrain.shape[0]              #size of training instances
    N_onion = count(ytrain)                 #num onion articles in train
    N_econ = doc_size - N_onion             #num econ articles in train
    vocabulary_size = xtrain.shape[1]       #size of vocabulary
    logprior = [0]*2
    logprior[0] = math.log(N_econ/float(doc_size)) #log of econ probability
    logprior[1] = math.log(N_onion/float(doc_size))#log of onion probability

    #loglikelihood = [[0]*vocabulary_size for i in range(2)] #loglikelihood for both classes
    #word_counts = [[0]*vocabulary_size for i in range(2)]#count for econ class
    loglikelihood = np.zeros((2,vocabulary_size))
    word_counts = np.zeros((2,vocabulary_size))



    for doc in range(doc_size):                   #get the counts for each word in each class
        for word in range(vocabulary_size):
            if ytrain[doc] == 0.0:    #we have an econ
               word_counts[0][word] = word_counts[0][word] + xtrain[doc][word]
            elif ytrain[doc] == 1.0: #we have an onion
                word_counts[1][word] = word_counts[1][word] + xtrain[doc][word]

    for article in range(2):    #for onion,econ
        for word in range(vocabulary_size): #for every word in vocab
            loglikelihood[article][word] = math.log((word_counts[article][word]+.5)/float((word_counts[article]).sum()+.5*vocabulary_size))

    return logprior, loglikelihood

 
def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    '''The function to run your ML algorithm on given datasets, generate the predictions and save them into the provided file path
     
    Parameters
    ----------
    Xtrain_file: string
        the path to Xtrain csv file
    Ytrain_file: string
        the path to Ytrain csv file
    test_data_file: string
        the path to test data csv file
    pred_file: string
        the prediction file to be saved by your code. You have to save your predictions into this file path following the same format of Ytrain_file
    '''
 
    ## your implementation here
    # read data from Xtrain_file, Ytrain_file and test_data_file
    ytrain = readData(Ytrain_file)
    xtrain = readData(Xtrain_file)
    test_data = readData(test_data_file) 

    #test_size = (int)((0.1)*ytrain.shape[0])
    #test_data = np.zeros((50,xtrain.shape[1]))
    #training_size = (int)(450*1)
    #for test_doc in range(test_size):
    #    test_data[test_doc] = xtrain[450+test_doc]
    #correct_predictions = ytrain[450:500]
    #xtrain = xtrain[0:training_size]
    #ytrain = ytrain[0:training_size]

    (logprior, loglikelihood) = trainBayes(xtrain,ytrain)

    predictions = np.zeros((test_data.shape[0],1))

    for doc in range(test_data.shape[0]): #for each document in test file
        curr_sum = np.copy(logprior)
        for article in range(2):
            for word in range(test_data.shape[1]):
                curr_sum[article] = curr_sum[article] + loglikelihood[article][word]*test_data[doc][word]
        if(curr_sum[0] > curr_sum[1]):
            predictions[doc] = 0
        else:
            predictions[doc] = 1
    predictions = predictions.astype(int)

    #correct = 0
    #for i in range(50):
    #    if(predictions[i] == correct_predictions[i]):
    #        correct = correct + 1
    #print "ACCURACY: ", (correct/50.0)


    np.savetxt(pred_file, predictions, fmt = '%d', delimiter=',')
 
    # save your predictions into the file pred_file
 
 
# define other functions here
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Received wrong number of arguments. \nUsage: \'python run.py xtrain_file ytrain_file test_data_file prediction_file ")
        sys.exit()
    run(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])