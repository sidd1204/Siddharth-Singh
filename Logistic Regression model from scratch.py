import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv("ex2data1.txt", header= None, names=["Exam1","Exam2","Admitted"])


def sigmoid(z):
    return 1/(1+np.exp(-z))

data.insert(0,"ones",1)
data.head()
cols = data.shape[1]  
x= data.iloc[:,0:cols-1]  
y= data.iloc[:,cols-1:cols]

X=np.array(x.values)
Y=np.array(y.values)
theta = np.zeros(3)

def cost(theta,X,Y):
    X = np.matrix(X)
    Y = np.matrix(Y)
    theta=np.matrix(theta)
    
    first= np.multiply(-Y, np.log(sigmoid(X*theta.T)))
    second= np.multiply((1-Y), np.log(1 - (sigmoid(X*theta.T))))
    return np.sum(first-second) /len(X)

J=cost(theta,X,Y)
J

def gradient(theta,X,Y):
    X = np.matrix(X)
    Y = np.matrix(Y)
    theta=np.matrix(theta)
    parameters= len(theta.T)
    error= sigmoid(X*theta.T) - Y
    grad= np.zeros(parameters)
    for i in range(parameters):
        term= np.multiply(error,X[:,i])
        grad[i]= np.sum(term) / len(X)
    return grad

#new theta "g"
g=gradient(theta,X,Y)
g    

# Optimizing the theta "g" 
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X,Y))
result
cost(result[0], X, Y)



def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]    

theta_min= np.matrix(result[0])
predictions = predict(theta_min, X) 
predictions


from sklearn.metrics import confusion_matrix
confusion_matrix(Y, predictions)


