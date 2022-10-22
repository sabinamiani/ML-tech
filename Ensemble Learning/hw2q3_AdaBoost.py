"""
@author Sabina Miani
@date October 2022
@assignment CS 5350 - HW2 
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd

def AdaBoost(X, y, T = 500) : 
    m = X.shape[0]
    # int D1(i) = 1/m
    Dt = np.ones(m) / m

    Hf = np.zeros(m)
    
    for t in np.arange(T) : 
        # find ht - decision stump 
        ht = h_stump(X,y,Dt)
        ht_predict = predict(X,ht)

        # calc error
        w_error = np.sum(Dt * np.not_equal(y, ht_predict))

        # calc alpha_t
        alpha = 1/2 * np.log((1-w_error)/ w_error)
 
        # calc new weight 
        Dt = Dt * np.exp(-alpha * np.not_equal(y, ht_predict))
    
        # instead of saving values, just sum for each t
        Hf += alpha*ht_predict
    # fix sign
    Hf = np.sign(Hf)

    return Hf

def h_stump(X,y,Dt) :
    ht = {}

    return ht

def predict(X,ht) :

    return []


# Parses csv file 
def csv_to_data(CSVfile):
    # turn into 2d array of data points  
    data = []
    with open(CSVfile , 'r') as f :
        for line in f :
            # terms = one row of data = one data point
            terms = line.strip().split(',')
            data.append(terms)

    return np.array(data)


### bank data and features ###
# 16 features 
bank_features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
bank_feats_numrd = np.array([np.arange(1,17), bank_features]).T
print(bank_feats_numrd)

# training and testing values from the bank csv files
training_data = csv_to_data('bank/train.csv')
testing_data = csv_to_data('bank/test.csv')

# all but the last column 
X_train = training_data[:,:-1]
X_test = testing_data[:,:-1]
# print(X_train.shape)
# only the last column 
y_train = training_data[:,-1]
y_test = testing_data[:,-1]
# print(y_train.shape)


# run Adaboost on the training data 
Hf = AdaBoost(X_train, y_train)