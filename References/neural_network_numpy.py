# Neural network using only numpy!
# With help from Siraj Raval
# --> https://www.youtube.com/watch?v=262XJe2I2D0&t=1471s

import numpy as np

def sigmoid(x, deriv=False):
    if(deriv==True):
        return x * (1 - x)
    
    return 1/(1 + np.exp(-x))
#input data
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#output (labels)
y = np.array([[0],
              [1],
              [1],
              [0]])
np.random.seed(0)

#synapses
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

#training
num_epochs = 10 ** 5
for i in range(num_epochs):
    #layers
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    #backpropagation
    l2_error = y - l2
    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = np.dot(l2_delta, syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

    if (i % 10000) == 0:
        print('Cost: ' + str(np.mean(np.absolute(l2_error))))

print('Output after training:')
print(l2)
