import numpy as np

x1 = np.array([1, 1, -1, 1, 3])
x2 = np.array([-1,1,1,2,-1])
x3 = np.array([2,3,0,-4,-1])
x = np.array([x1, x2, x3]).T

y  = np.array([1,4,-1,-2,0]).T
w = np.array([-1,1,-1])
b = -1

db = y - np.dot(x,w) - b
dw = x.T.dot(db) / 5
db = 1/5 * db
print('dw is ',dw)
print('db is ', db)

w = np.zeros(4)
lr = 0.1
x = np.array([x1, x2, x3, np.ones(5)]).T
db = 0
dw = np.zeros(4)

for i in np.arange(5):
    ins = y[i] - np.dot(x[i],w)
    dw += x[i].T.dot(ins) / (i + 1)
    print('stochastic gradient at example ', i + 1, ' is ', dw)

    w = w - lr * dw

    print('[w1, w2, w3, b] = ', w)