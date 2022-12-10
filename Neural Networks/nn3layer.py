"""
@author Sabina Miani
@date December 2022
@assignment CS 5350 - HW5 - Neural Networks
"""

import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

### backpropogation algorithm 
def threeLayerNN(X, y, weights, width=3, num_hidden=2) :
    W1 = weights['W1']
    W2 = weights['W2']
    if num_hidden == 2 :
        W3 = weights['W3']

    # forward pass 
    # account for bias term
    h_l1 = np.concatenate(([1],sigmoid(np.dot(W1, X))))
    print('h_l1=', h_l1)
    
    ystar = 0
    if num_hidden == 2 :
        h_l2 = np.concatenate(([1],sigmoid(np.dot(W2, h_l1))))
        print('h_l2=', h_l2)
        ystar = np.dot(h_l2, W3)

    elif num_hidden == 1 :
        ystar = np.dot(h_l1, W2)

    print('predicted y = ', ystar) 

    # loss 
    loss = 0.5 * np.square(y - ystar)

    # backwards pass 
    grads = {}

    dloss = y - ystar

    # hidden layers first
    if num_hidden == 2 : 
        grads['W3'] = np.dot(dloss, h_l2) 
        print('w3=', grads['W3'])

        # backpropogate
        dh_l2_1 = dloss * W3[1] * dsigmoid(h_l2[1])
        dh_l2_2 = dloss * W3[2] * dsigmoid(h_l2[2])
        # print('multipliers', dloss, W3[1], dsigmoid(h_l2[1]), h_l2[1])

        grads['W2_1'] = np.dot(dh_l2_1, h_l1)
        print('w2_1=', grads['W2_1'])
        grads['W2_2'] = np.dot(dh_l2_2, h_l1)
        print('w2_2=', grads['W2_2'])

        # backpropogate
        dh_l1_1 = (dh_l2_1*W2[0,1] + dh_l2_2*W2[1,1]) * dsigmoid(h_l1[1])
        dh_l1_2 = (dh_l2_1*W2[0,2] + dh_l2_2*W2[1,2]) * dsigmoid(h_l1[2])

    elif num_hidden == 1 :
        grads['W2'] = np.dot(dloss, h_l1) 
        print('w2=', grads['W2'])

        # backpropogate
        dh_l1_1 = dloss * W2[1] * dsigmoid(h_l1[1])
        dh_l1_2 = dloss * W2[2] * dsigmoid(h_l1[2])
        
    # first layer 
    grads['W1_1'] = np.dot(dh_l1_1, X)
    print('w1_1=', grads['W1_1'])
    grads['W1_2'] = np.dot(dh_l1_2, X)
    print('w1_2=', grads['W1_2'])

    return grads


# 3 layer nn testing with single x, y point and fixed weights 
X = np.array([1,1,1])
y = 1
weights = {
    'W1': np.array([[-1,-2,-3],[1,2,3]]), 
    'W2': np.array([[-1,-2,-3],[1,2,3]]), 
    'W3': np.array([-1,2,-1.5])
    }

grads = threeLayerNN(X,y,weights)
print(grads)


# sgd implementation using backpropogation 
def lr_schedule(lr, d, t) :
    return lr / (1 + lr/d*t)

def sgd_nn(X,y,lr,d,width,T=100,num_hidden=2,zeros=False) :
    # init w
    num_train, num_features = np.shape(X)

    if zeros :
        W = np.zeros(num_features) 
    else :
        W = np.random.normal(size=(num_features))

    for t in range(T) :
        # shuffle 
        rand_indices = np.random.choice(num_train, num_train, replace=False)
        X_batch = X[rand_indices]
        y_batch = y[rand_indices]

        # run backprop alg to get dL 
        dL = threeLayerNN(X_batch, y_batch, W, width, num_hidden)

        # update W
        W = W - lr*dL

    # update lr
    lr = lr_schedule(lr, d, t)

    return W 

def accuracyCheck(X,y,W) :

    return 

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

# change y data from {0,1} to {-1,1} 
for i in range(y_train.shape[0]) :
    if y_train[i] == 0 :
        y_train[i] = -1
# print(y_train)
for i in range(y_test.shape[0]) :
    if y_test[i] == 0 :
        y_test[i] = -1
# print(y_test)

# augment X with bias term 
bias = np.ones(X_train.shape[0])
bias.shape = [X_train.shape[0],1]
X_train = np.append(X_train, bias, axis=1)

bias = np.ones(X_test.shape[0])
bias.shape = [X_test.shape[0],1]
X_test = np.append(X_test, bias, axis=1)


widths = [5,10,25,50,100]
lr = 0.1
d = 1

for width in widths :
    print('width: ', width)

    print('init W to gaussian dist')
    W = sgd_nn(X_train,y_train,lr,d,width)

    print('training set accuracy: ')
    accuracyCheck(X_train,y_train,W)
    print('testing set accuracy: ')
    accuracyCheck(X_test,y_test,W)

    print('init W to zeros')
    W = sgd_nn(X_train,y_train,lr,d,width,zeros=True)

    print('training set accuracy: ')
    accuracyCheck(X_train,y_train,W)
    print('testing set accuracy: ')
    accuracyCheck(X_test,y_test,W)
