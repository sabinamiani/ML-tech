"""
@author Sabina Miani
@date December 2022
@assignment CS 5350 - HW5 - Neural Networks -- BONUS Questions 
"""

import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def NNPyTorch(X,Y, depth, width, activation, T=1) :

    # create model using torch sequential 
    # FC linear layers with specified activation function 
    layers = []

    # first layer
    layers.append(nn.Linear(X.shape[1], width)) 
    # activation 
    if activation == 'relu' :
        layers.append(nn.ReLU())
    elif activation == 'tanh' :
        layers.append(nn.Tanh())

    for d in range(depth - 2) :
        # hidden layers
        layers.append(nn.Linear(width, width))

        # activation 
        if activation == 'relu' :
            layers.append(nn.ReLU())
        elif activation == 'tanh' :
            layers.append(nn.Tanh())
    
    # last layer
    layers.append(nn.Linear(width, width)) 

    model = nn.Sequential(*layers)
    # print(model)

    # optimize using adam 
    optimizer = optim.Adam(model.parameters())

    # for each epoch iterate over all examples of x,y and train model 
    for epoch in range(T) :
        for i in range(X.shape[0]) :
            x = X[i]
            y = Y[i]
            # print(x, y)

            # train model
            model.train()

            # compute loss 
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # backprop w optimizer step 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print current results 
            # if i % 200 == 0:
            #     print('Iteration %d, loss = %.4f' % (i, loss.item()))
            #     accuracyCheck(model,X,Y)
            #     print()

    return model


def accuracyCheck(model,x,y) :
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



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

# get X,y from bank-note csv files 
# 4 attributes + last column label (genuine or forged)
attributes = ['variance', 'skewness', 'curtosis', 'entropy']

# training and testing values from the concrete csv files
training_data = csv_to_data('bank-note/train.csv')
testing_data = csv_to_data('bank-note/test.csv')

# all but the last column 
X_train = training_data[:,:-1]
X_test = testing_data[:,:-1]
# only the last column 
y_train = training_data[:,-1]
y_test = testing_data[:,-1]

# convert to tensors 
X_train = torch.Tensor(X_train)
# print(X_train, X_train.size())
y_train = torch.LongTensor(y_train)
# print(y_train, y_train.size())
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)

depths = [3,5,9]
widths = [5,10,25,50,100]
acts = ['relu', 'tanh']

for depth in depths :
    print('depth: ', depth)
    for width in widths : 
        print('width: ', width)
        for act in acts :
            print('activation: ', act)

            model = NNPyTorch(X_train,y_train,depth,width,act)

            print('training set accuracy: ')
            accuracyCheck(model, X_train, y_train)

            print('testing set accuracy: ')
            accuracyCheck(model, X_test, y_test)

            print()
