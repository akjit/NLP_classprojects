#!/usr/bin/env python
import numpy as np 
import sys
import math
import time


def readData(filename):
    datafile = open(filename, 'r')
    data = np.matrix(np.genfromtxt(filename,delimiter = ","))
    datafile.close()
    return data

def trainPerceptron(xtrain, ytrain, epochs=5):
    k = 0
    c = [0]
    w = np.matrix(np.zeros((1,xtrain.shape[1])))

    sign = 0
    for epoch in range(epochs):
        for doc in range(ytrain.shape[0]):
            sign = ytrain[doc][0]
            if sign * np.dot(w[k],xtrain[doc].T) <= 0:
                neww = w[k] + sign * xtrain[doc]
                w = np.vstack((w,neww))
                c.append(1)
                k = k + 1
            else:
                c[k] = c[k] + 1
    return w,c





 
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
    xtrain = readData(Xtrain_file)
    ytrain = readData(Ytrain_file)
    ytrain = np.copy(ytrain.T)
    for i in range(ytrain.shape[0]):
        if ytrain[i] == 0:
            ytrain[i] = -1
    test_data = readData(test_data_file)
    predictions = np.zeros((test_data.shape[0],1))

    #test_size = (int)((0.1)*ytrain.shape[0])
    #test_data = np.zeros((50,xtrain.shape[1]))
    #training_size = (int)(450*.05)
    #for test_doc in range(test_size):
    #    test_data[test_doc] = xtrain[450+test_doc]
    #correct_predictions = ytrain[450:500]
    #xtrain = xtrain[0:training_size]
    #ytrain = ytrain[0:training_size]

    (weights,count) = trainPerceptron(xtrain,ytrain)
    final_prediction = 0
    for doc in range(test_data.shape[0]):
        for i in range(len(count)):
           final_prediction = final_prediction + count[i]*np.sign(np.dot(weights[i],test_data[doc].T))
        if final_prediction > 0:
            predictions[doc] = 1
        else:
            predictions[doc] = 0
        final_prediction = 0

    #correct = 0
    #for i in range(50):
    #    if((predictions[i] == 1 and correct_predictions[i] == 1 ) or (predictions[i] == 0 and correct_predictions[i] == -1)):
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