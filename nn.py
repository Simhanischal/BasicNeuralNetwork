import numpy as np
np.random.seed(25)

def sigmoid(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,1,1],[1,0,1],[0,0,1],[1,1,0]])
Y = np.array([[1,1,0,0]]).T
weight1 = 2*np.random.random((3,4))-1
weight2 = 2*np.random.random((4,1))-1

for j in range(50000):
    layer1 = X
    layer2 = sigmoid(np.dot(layer1,weight1))
    layer3 = sigmoid(np.dot(layer2,weight2))
    l3_error = Y - layer3
    if j%10000==0:
        print("Error is "+str(np.mean(np.abs(l3_error))))
    l3_del = l3_error * sigmoid(layer3,deriv=True)
    l2_error = l3_del.dot(weight2.T)
    l2_del = l2_error * sigmoid(layer2,deriv=True)
    weight2 += (layer2.T).dot(l3_del)  
    weight1 += (layer1.T).dot(l2_del)
print(layer3)  