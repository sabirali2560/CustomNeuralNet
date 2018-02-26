import numpy as np

m,n=np.shape(X)   # X be the given input

# initializing the weights and biases

W1=np.random.random((hid_layer0,n))  #hid_layer0 be the size of the 1st hidden layer
b1=np.random.random((hid_layer0,1))
W2=np.random.random((hid_layer1,hid_layer0))  #hid_layer1 be the size of the 2nd hidden layer
b2=np.random.random((hid_layer1,1))
W3=np.random.random((num_out,hid_layer1))     #num_out be the size of the output layer
b3=np.random.random((num_out,1))


# sigmoid activation function

def sigmoid(z):
    a=(np.ones((np.shape(z))))/((np.ones((np.shape(z))))+np.exp((-1)*(z)))
    return a

# implementing front propagation

z1=np.dot(X,W1.T)+b1.T
a1=sigmoid(z1)
z2=np.dot(a1,W2.T)+b2.T
a2=sigmoid(z2)
z3=np.dot(a2,W3.T)+b3.T
a3=sigmoid(z3)

# costfunction

def costfunc (a3,W1,W2,W3,y,m,lam):     # lam is the regularization parameter       # y is the given output
    sum0=-np.sum((np.dot(y,np.log(a3).T)))-np.sum((np.dot((np.ones((np.shape(y)))-y),(np.ones((np.shape(a3)))-np.log(a3)).T)))
    t1=np.sum(W1**2)
    t2=np.sum(W2**2)
    t3=np.sum(W3**2)
    sum1=t1+t2+t3
    cost=((1/m)*(sum0))+(((lam)/(2*m))*(sum1))
    return cost

# implementing backpropagation

def gradient(X,a1,a2,a3,y,W1,W2,W3,b1,b2,b3,lam):
    Del1=np.zeros((np.shape(W1)))
    Del2=np.zeros((np.shape(W2)))
    Del3 = np.zeros((np.shape(W3)))
    Del10 = np.zeros((np.shape(b1)))
    Del20 = np.zeros((np.shape(b2)))
    Del30 = np.zeros((np.shape(b3)))
    for i in range(0,m):
        A1=np.array([X[i,:]]).T
        A2=np.array([a1[i,:]]).T
        A3=np.array([a2[i,:]]).T
        A4=np.array([a3[i,:]]).T
        y0=np.array([y[i,:]]).T
        del4=A4-y0
        del3=np.dot(W3.T,del4)*(A3)*(np.ones((np.shape(A3)))-A3)
        del2 = np.dot(W2.T, del3) * (A2) * (np.ones((np.shape(A2))) - A2)
        Del1=Del1+np.dot(np.array([del2]).T,A1.T)
        Del2=Del2+np.dot(np.array([del3]).T,A2.T)
        Del3 = Del3 + np.dot(np.array([del4]).T, A3.T)
        Del10= Del10 + del2
        Del20 = Del20 + del3
        Del30 = Del30 + del4
    W1_grad = (1 / np.shape(X)[0]) * (Del1 + lam * W1)
    W2_grad = (1 / np.shape(X)[0]) * (Del2 + lam * W2)
    W3_grad = (1 / np.shape(X)[0]) * (Del3 + lam * W3)
    b1_grad = (1 / np.shape(X)[0]) * (Del10)
    b2_grad = (1 / np.shape(X)[0]) * (Del20)
    b3_grad = (1 / np.shape(X)[0]) * (Del30)
    return W1_grad,W2_grad,W3_grad,b1_grad,b2_grad,b3_grad

