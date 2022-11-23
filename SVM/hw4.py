"""
@author Sabina Miani
@date November 2022
@assignment CS 5350 - HW4 - SVMs
"""

import numpy as np 
from matplotlib import pyplot as plt


def primal_svm(X,y,C,l0,lrf, a=1, T=100) :
    num_train, num_features = np.shape(X)
    # init w
    W = np.zeros(num_features, dtype=float)
    # print(np.size(W))
    obj = []

    # T epochs
    for t in range(T):
        # shuffle data 
        shuffled_is = np.random.choice(num_train, num_train, replace=False)
        
        # for each training example
        for i in shuffled_is :
            sub_grad = y[i] * np.dot(X[i], W)
            # print(sub_grad)
            if sub_grad <= 1:
                # update w
                W0 = W.copy()
                W0[-1] = 0
                W = (1-l0)*W0 + l0*C*num_train*(y[i] * X[i])

            else :
                W[:-1] = (1-l0)*W[:-1]

        # objective function
        # hinge loss
        distances = 1 - y * (np.dot(X, W))
        distances[distances < 0] = 0 
        loss = C * (np.sum(distances) /  X.shape[0])
        obj.append(1 / 2 * np.dot(W, W) + loss)
        # print(np.shape(obj))

        # update the learning rate after each step 
        l0 = lrf(l0,a,t)
        # print(l0)

    return W, obj

# return avg prediction error using given w,X,y for the primal svm alg
def primal_svm_predict(W, X, y) :
    prediction = np.sign(np.dot(X, W))
    error = np.sum(np.abs(prediction - y)) / y.shape[0]
    
    return error


# def dual_svm () :
    
#     W = 0

#     return W 


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



print('\nCS5350 HW4 - SVMs implemented by Sabina Miani')

T = 100 
C = np.array([100/873, 500/873, 700/873]) 

# (2a) primal SVM -> lr = lr/(1+(lr/a*t))
print('\nPrimal SVM (a)')

lr = 0.001
a = 0.001
print('lr=', lr, ' a=', a)

def lrfa(l0,a,t) :
    return l0 / (1+(l0/a*t))

for c in C :
    print('C=', c)

    w, obj = primal_svm(X_train,y_train,c,lr,lrfa,a,T)
    
    plt.plot(range(T), obj)
    plt.show()

    print('[w0,b] =', w)

    # report training error
    p_train = primal_svm_predict(w, X_train, y_train)
    print('training error =',p_train)
    # report test error
    p_test = primal_svm_predict(w, X_test, y_test)
    print('test error =',p_test)


# (2b) primal SVM -> lr = lr/(1+t)
print('\nPrimal SVM (b)')

lr = 0.001
a = 0.001
print('lr=', lr, ' a=', a)

def lrfb(l0,a,t) :
    return l0 / (1+t)

for c in C :
    print('C=', c)

    w, obj = primal_svm(X_train,y_train,c,lr,lrfb,a,T)
    
    plt.plot(range(T), obj)
    plt.show()

    print('[w0,b] =', w)

    # report training error
    p_train = primal_svm_predict(w, X_train, y_train)
    print('training error =',p_train)
    # report test error
    p_test = primal_svm_predict(w, X_test, y_test)
    print('test error =',p_test)


# (3) dual SVM 
# print('\Dual SVM (a)')


