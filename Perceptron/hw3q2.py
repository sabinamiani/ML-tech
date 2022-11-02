"""
@author Sabina Miani
@date November 2022
@assignment CS 5350 - HW3 - Perceptrons
"""

import numpy as np 


# (a) standard perceptron 
#     T = 10 
#     return learned weight vector, avg test prediction error 
def std_perceptron(X, y, lr, T = 10):
    num_train, num_features = np.shape(X)
    # init w
    W = np.zeros(num_features, dtype=float)

    # T epochs
    for t in range(T):
        # shuffle data 
        shuffled_is = np.random.choice(num_train, num_train, replace=False)
        
        # for each training example
        for i in shuffled_is :
            if np.dot(y[i],np.dot(X[i], W)) <= 0:
                # update w
                W = W + lr*(np.dot(y[i],X[i]))

    return W

# return avg test prediction error using given w,X,y for the standard perceptron alg
def std_predict(W, X, y) :
    prediction = np.sign(np.dot(X, W))
    error = np.sum(np.abs(prediction - y)) / y.shape[0]
    
    return error


# (b) voted perceptron 
#     T = 10 
#     return list of distinct weight vectors and their counts, 
#       avg test prediction error using all distinct weight vectors 
def voted_perceptron(X, y, lr, T = 10):
    num_train, num_features = np.shape(X)
    # init w and m
    W = np.zeros(num_features, dtype=float)
    weights = {}
    counts = {}
    m = 0
    cm = 0

    # T epochs
    for t in range(T):
        # shuffle data 
        shuffled_is = np.random.choice(num_train, num_train, replace=False)
        
        # for each training example
        for i in shuffled_is :
            if np.dot(y[i],np.dot(X[i], W)) <= 0:
                # add unique W, cm to dicts
                weights[m] = W
                counts[m] = cm

                # update w
                W = W + lr*(np.dot(y[i],X[i]))
                
                # special voted updates 
                m += 1
                cm = 1

            else :
                cm = cm + 1

    return weights, counts

# return avg test prediction error using given w,X,y for the voted perceptron alg
def voted_predict(weights, counts, X, y) :
    sum = 0
    for i in range(1, len(weights)):
        sum += counts[i] * np.sign(np.dot(X, weights[i]))
    prediction = np.sign(sum)
    error = np.sum(np.abs(prediction - y)) / y.shape[0]
    
    return error


# (c) average perceptron 
#     T = 10 
#     return learned weight vector, avg test prediction error 
def avg_perceptron(X, y, lr, T = 10):
    num_train, num_features = np.shape(X)
    # init w and a
    W = np.zeros(num_features, dtype=float)
    a = np.zeros(num_features, dtype=float)

    # T epochs
    for t in range(T):
        # shuffle data 
        shuffled_is = np.random.choice(num_train, num_train, replace=False)
        
        # for each training example
        for i in shuffled_is :
            if np.dot(y[i],np.dot(X[i], W)) <= 0:
                # update w
                W = W + lr*(np.dot(y[i],X[i]))
            a = a + W

    return a

# return avg test prediction error using given w,X,y for the average perceptron alg
def avg_predict(a, X, y) :
    prediction = np.sign(np.dot(X, a))
    error = np.sum(np.abs(prediction - y)) / y.shape[0]
    
    return error


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

# change y data from {0,1} to {-1,1} for easier perceptron implementation 
for i in range(y_train.shape[0]) :
    if y_train[i] == 0 :
        y_train[i] = -1
# print(y_train)
for i in range(y_test.shape[0]) :
    if y_test[i] == 0 :
        y_test[i] = -1
# print(y_test)


print('\nCS5350 HW3 - Perceptrons implemented by Sabina Miani')
learning_rate = 0.1

# (a) - std
print('\n(a) Standard Perceptron')
# learning_rate = 0.1
w = std_perceptron(X_train, y_train, learning_rate, T = 10)
print('w = ', w)
# std_error = std_predict(w, X_train, y_train)
# print(std_error)
std_error = std_predict(w, X_test, y_test)
print('test error: ', std_error)


# (b) - voted
print('\n(b) Voted Perceptron')
# learning_rate = 0.1
weights, counts = voted_perceptron(X_train, y_train, learning_rate, T = 10)
# print weights and counts
for i in range(1, len(weights)) :
    print(i, '- c=', counts[i], ', w=', weights[i])
vo_error = voted_predict(weights, counts, X_test, y_test)
print('test error: ', vo_error)

# summation of counts*weights
weights = np.array(list(weights.items()))
counts = np.array(list(counts.items()))
print(np.sum(counts*weights, axis=0))


# (c) - avg
print('\n(c) Average Perceptron')
# learning_rate = 0.1
a = avg_perceptron(X_train, y_train, learning_rate, T = 10)
print('a = ', a)
# avg_error = avg_predict(a, X_train, y_train)
# print(avg_error)
avg_error = avg_predict(a, X_test, y_test)
print('test error: ', avg_error)
