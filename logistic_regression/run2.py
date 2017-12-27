#!/usr/bin/env python
 
# import the required packages here
from __future__ import print_function
import os
import sys
import math
import numpy as np


def readData(filename):
    datafile = open(filename, 'r')
    data = np.matrix(np.genfromtxt(filename,delimiter = ","))
    datafile.close()
    return data

def sigmoid(weights,document):
    data = np.dot(document,weights)
    g = 1.0 / (1.0 + np.exp(-1 * data))
    return g

def LCL(training,weights):
    total = 0.0
    for i in range(training.shape[0]):
        y = training[i,training.shape[1]-1]
        x = training[i,0:training.shape[1]-1]
        p = sigmoid(weights,x)
        if p == 1:
            p = p-0.0000001
        if p == 0:
            p = p+0.0000001
        total = total + y * math.log(p) + (1-y)*math.log(1-p)

    return total


def gradiant_descent(training,weights,lr,lr_d,l2_d,iterations=1):
    threshold = .0001          #threshold value for convergence
    converged = False
    for it in range(iterations):
        subber = l2_d * np.square(weights).sum()
        lcl = LCL(training,weights)
        #print("LCL: " + str(lcl) + " SUB: " + str(subber))
        LCL_L2 = lcl - subber       #Calculate Initial LCL
        while not converged:                                                    #While not converged (LCL - NewLCL is not small)
            np.random.shuffle(training)                                         #shuffle data
            for i in range(training.shape[0]):                                  #for every training document
                y = training[i,training.shape[1]-1]                             #isolate y label
                x = training[i,0:training.shape[1]-1]                           #remove label from training
                p = sigmoid(weights,x)                                          #calculate sigmoid with weights for this doc
                first = (y-p)*x                                                 #(y-p) are both ints; *x to affect training as a scalar
                second = 2*l2_d*weights                                         #subtract weights * l2 decay
                adder = lr * (first.T - second)                                 #multiply the sum of the above by learning rate
                weights = weights + adder                                       #(inc/dec)remement weights
            #print("OLDLCL: " + str(LCL_L2),end='')
            subber = l2_d * np.square(weights).sum()
            lcl = LCL(training,weights)                                         
            newLCL = lcl - subber                                               #after training on all documents, calculate newLCL
            #print(" NEWLCL: " + str(lcl) + " SUB: " + str(subber))
            if(abs(LCL_L2 - newLCL) < threshold):                               #if difference is < threshold, return
                converged = True
            else:                                                               #else set oldLCL to value in newLCL, decay lr, and continue iterating
                LCL_L2 = newLCL
                lr = lr * lr_d
        # if(it + 1 != iterations):
        #     lr = lr * lr_d 
        #     converged = False
    return weights


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

    xtrain = readData(Xtrain_file)
    ytrain = readData(Ytrain_file)
    test_data = readData(test_data_file)
    ytrain = np.copy(ytrain.T)
    training = np.c_[xtrain,ytrain]               #concatenate the ytrain with the end of each corresponding xtrain
    #weights = np.random.rand(training.shape[0],1)
    init_weights = np.random.rand(training.shape[0],1)   #initialize random weights
    #np.random.shuffle(training)                     #shuffle the document order

    learning_rate = .855                       #initial learning rate
    lr_decay = 0.95                               #learning rate decay value
    l2_decay = [0, .0000001, .000001, .00001, .0001, .001, .01, .1, 1]                            #l2 regularization lambda constant
    #l2_decay = .001517

    # learning_rate = [0.2, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]
    # lr_decay = [.001,.01,.5,.7,.8,.85,.9]
    # l2_decay = [.0000035,.00001,.0001, .001, .01, .1, .5]

    # top_accur = [0]*10
    # top_lr    = [0]*10
    # top_lrd   = [0]*10
    # top_l2d   = [0]*10

    for i in range(10):
        size = training.shape[0]
        train_percent = training[size*(i/10.0):size*((i+1)/10.0)]     #train over some percent of the data
        test_percent = np.r_[training[size*0:size*(i/10.0)],training[size*((i+1)/10.0):size*1]]
        test_pred = np.r_[ytrain[size*0:size*(i/10.0)],ytrain[size*((i+1)/10.0):size*1]]
        for regularization in range(9):
    #        for lrd_iter in range(7):
    #           for l2d_iter in range(7):
    #             weights = init_weights
    #             weights = gradiant_descent(train_percent,weights,learning_rate[lr_iter],lr_decay[lrd_iter],l2_decay[l2d_iter])
            weights = init_weights
            weights = gradiant_descent(train_percent,weights,learning_rate,lr_decay,l2_decay[regularization])

            predictions = np.zeros((test_percent.shape[0],1))

            # print(test_percent.shape)
            # print(test_pred.shape)

            for tester in range(test_percent.shape[0]):
                p = sigmoid(weights,test_percent[tester,0:test_percent.shape[1]-1])
                if(p > (1-p)):
                    predictions[tester] = 1
                else:
                    predictions[tester] = 0

            #np.savetxt(pred_file,predictions, fmt = '%d', delimiter = ',')

            correct = 0.0
            for guess in range(test_pred.shape[0]):
                if predictions[guess] == test_pred[guess]:
                    correct = correct + 1.0

            accuracy = correct/(test_pred.shape[0])
            print("FOLD: " + str(i+1) + " Accuracy: " + str(accuracy) + " L2D : " + str(l2_decay[regularization]))

        #         print("FOLD: " + str(i+1) + " Accuracy: " + str(accuracy) + " LR: " + str(learning_rate[lr_iter]) + " LRD: " + str(lr_decay[lrd_iter]) + " L2D: " + str(l2_decay[l2d_iter]))
        #         if accuracy > top_accur[i]:
        #            top_accur[i] = accuracy
        #            top_lr[i] = learning_rate[lr_iter]
        #            top_lrd[i] = lr_decay[lrd_iter]
        #            top_l2d[i] = l2_decay[l2d_iter]
        # print("\n\n\n\nCV: " + str(i+1) + " Accuracy: " + str(top_accur[i]) + " LR: " + str(top_lr[i]) + " LRD: " + str(top_lrd[i]) + " L2D: " + str(top_l2d[i]) + "\n\n\n\n")

    # for i in range(10):
    #    print("\nCV: " + str(i+1) + " Accuracy: " + str(top_accur[i]) + " LR: " + str(top_lr[i]) + " LRD: " + str(top_lrd[i]) + " L2D: " + str(top_l2d[i]))



    
    ## your implementation here
    # read data from Xtrain_file, Ytrain_file and test_data_file
 
    # your algorithm
 
    # save your predictions into the file pred_file
 
 
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Received wrong number of arguments. \nUsage: \'python run.py xtrain_file ytrain_file test_data_file prediction_file ")
        sys.exit()
    run(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])