"""
@author Sabina Miani
@date October 2022
@assignment CS 5350 - HW2 
"""

import numpy as np
from numpy import linalg as LA


def batch_GD(X, y, learning_rate, iters) : 
    num_train, num_features = np.shape(X)
    W = np.zeros(num_features, dtype=float)
    costs = {}

    for it in np.arange(iters) : 
        # compute gradient 
        scores = np.dot(X,W) 
        grad = np.dot(X.T, (scores - y)) / num_train

        # calc cost 
        costs[it] = np.sum(np.square(scores - y)) / num_train

        W_last = W
        W = W_last - (learning_rate * grad)
    
        # # norm of weight vector difference - ||w_t - w_t-1||
        # wnorm = LA.norm(W - W_last)
        # print(wnorm)

    return costs, W


def SGD(X, y, learning_rate, iters) : 
    num_train, num_features = np.shape(X)
    W = np.zeros(num_features, dtype=float)
    costs = {}

    for it in np.arange(iters) : 
        rand_indices = np.random.choice(num_train, num_train, replace=True)
        X_batch = X[rand_indices]
        y_batch = y[rand_indices]

        # compute gradient 
        scores = np.dot(X_batch,W) 
        grad = np.dot(X_batch.T, (scores - y_batch)) / num_train

        # calc cost 
        costs[it] = np.sum(np.square(scores - y)) / num_train

        W = W + (learning_rate * grad)

    return costs, W


# Parses csv file 
def csv_to_data(CSVfile):
    # turn into 2d array of data points  
    data = []
    with open(CSVfile , 'r') as f :
        for line in f :
            # terms = one row of data = one data point
            terms = line.strip().split(',')
            data.append(terms)

    return np.array(data, dtype=float)


### concrete data and features ###
# 7 features 
concrete_features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']

# training and testing values from the concrete csv files
training_data = csv_to_data('concrete/train.csv')
testing_data = csv_to_data('concrete/test.csv')

# all but the last column 
X_train = training_data[:,:-1]
X_test = testing_data[:,:-1]
# only the last column 
y_train = training_data[:,-1]
y_test = testing_data[:,-1]

# (a) run batch GD on concrete dataset 
iters = 500
learning_rate = 1e-1
costs, W = batch_GD(X_train, y_train, learning_rate, iters)
print('learned w with batch GD: ', W)
# print('costs: ', costs.values())

train_cost = np.sum(np.square(np.dot(X_train,W) - y_train)) / X_train.shape[0]
print('train cost using optimal weight: ', train_cost)

# cost w test data 
test_cost = np.sum(np.square(np.dot(X_test,W) - y_test)) / X_test.shape[0]
print('test cost using learned weight: ', test_cost)


# (b) run SGD on concrete dataset 
iters = 50
learning_rate = 1e-5
costs, W = SGD(X_train, y_train, learning_rate, iters)
print('learned w with SGD: ', W)
# print('costs: ', costs.values())

train_cost = np.sum(np.square(np.dot(X_train,W) - y_train)) / X_train.shape[0]
print('train cost using optimal weight: ', train_cost)

# cost w test data 
test_cost = np.sum(np.square(np.dot(X_test,W) - y_test)) / X_test.shape[0]
print('test cost using learned weight: ', test_cost)


# (c) analytical optimized weight vector 
X = X_train.T
Y = y_train
w_opt = LA.inv((X.dot(X.T))).dot(X).dot(Y)
print('optimal weight vector: ', w_opt)

train_cost = np.sum(np.square(np.dot(X_train,w_opt) - y_train)) / X_train.shape[0]
print('train cost using optimal weight: ', train_cost)

test_cost = np.sum(np.square(np.dot(X_test,w_opt) - y_test)) / X_test.shape[0]
print('test cost using optimal weight: ', test_cost)
