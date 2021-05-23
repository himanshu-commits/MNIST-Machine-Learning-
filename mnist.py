# importing the library
import numpy as np
import pandas as pd
import matplotlib
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#loading the dataset
data = pd.read_csv('mnist_train.csv')
#splitting the dataset
x = data.iloc[:,1:]
y = data.iloc[:,0]
print(x.shape)
print(y.shape)
x= np.array(x)
y = np.array(y)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.22)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
shuffle_training_sets=np.random.permutation(32760)
x_train,y_train=x_train[shuffle_training_sets],y_train[shuffle_training_sets]
#visualization of dataset
some_data = x_train[2010]
some_data_img = some_data.reshape(28,28)
plt.imshow(some_data_img ,cmap=matplotlib.cm.binary ,interpolation = "nearest")
some_data = x_train[2011]
some_data_img = some_data.reshape(28,28)
plt.imshow(some_data_img ,cmap=matplotlib.cm.binary ,interpolation = "nearest")
#reshaping the dataset
x_train = x_train.T
y_train = y_train.reshape(1,x_train.shape[1])
x_test = x_test.T
y_test = y_test.reshape(1,x_test.shape[1])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# Logistic Reression Model
def sigmoid(x):
    return 1/(1+np.exp(-x))
def model(x,y,iteration , learning_rate):
    m = x.shape[1]
    n = x.shape[0]
    theta = np.zeros((n,1))
    B = 0
    cost_list = []
    
    for i in range(iteration):
        Z = np.dot(theta.T,x)+B
        hypothesis = sigmoid(Z)
        cost = -(1/m)*np.sum(y*np.log(hypothesis)+(1 - y)*np.log(1 - hypothesis))
        dW = (1/m)*np.dot(hypothesis - y,x.T)
        dB = (1/m)*np.sum(hypothesis - y)
        theta = theta - learning_rate*dW.T
        B = B - learning_rate*dB
        cost_list.append(cost)
        if(i%(iteration/10)==0):
            print("cost", cost)
    return theta,B,cost_list
# Training the model
iteration = 10
learning_rate = 0.001
theta,B,cost_list = model(x_train , y_train , iteration = iteration,learning_rate = learning_rate)
# Testing the Model
plt.plot(np.arange(iteration),cost_list)
def accuracy(x,y,theta,b):
    z = np.dot(theta.T,x)+b
    prediction = sigmoid(z)
    acc = (1-np.sum(np.absolute(prediction - y))/y.shape[1])*100
    print("Accuracy of the model is : ",acc,"%")
accuracy(x_test,y_test,theta,B)s