import numpy as np 

X = np.array([[0.5, -1, 0.3,1],[-1,-2,-2,1],[1.5,0.2,-2.5,1]])
y = np.array([1,-1,1])
lr = np.array([0.01, 0.005, 0.0025])
T = 3

num_train, num_features = np.shape(X)
# init w
W = np.zeros(num_features, dtype=float)
# print(np.size(W))

# T epochs
for t in range(T):
    # shuffle data 
    shuffled_is = np.random.choice(num_train, num_train, replace=False)
    
    # for each training example
    # for i in shuffled_is :
    sub_grad = y * np.dot(X, W)
    print(sub_grad)
    if sub_grad.all() <= 1:
        # update w
        W0 = W.copy()
        W0[-1] = 0
        W = W - lr[t]*W0 + lr[t]*3*1/3*(np.dot(y,X))

    else :
        W[:-1] = (1-lr[t])*W[:-1]

