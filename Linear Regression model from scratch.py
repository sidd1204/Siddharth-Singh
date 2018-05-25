# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv("C:/Users/siddh/Desktop/DataScience/ex1data1.txt", header=None, names=['Population', 'Profit'])
print(data.describe())

x=data["Population"]
print(x)
y=data["Profit"]
print(y)

plt.title("DATA") 
plt.xlabel("Population") 
plt.ylabel("Profit") 
plt.scatter(x,y) 
plt.show()

data.insert(0,"Ones",1)
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

X = np.matrix(X) 
y = np.matrix(y)  
theta = np.matrix(np.array([0,0]))

# Defining Cost function
def Compute_Cost(X,y,theta):
    inner = np.power(((X * theta.T) - y.T), 2)
    return np.sum(inner) / (2 * len(X))

# Computing the Cost function
Compute_Cost(X,y,theta)

# Defining gradient descent to reduce the Cost funtion
def gradient_descent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    
    for i in range(iters):
        error= (X*theta.T)-y.T
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j]= theta[0,j] - ((alpha/len(X)) * np.sum(term))
        theta=temp
    cost= Compute_Cost(X,y,theta)
    return theta, cost

alpha= 0.01
iters= 1000

# Computing the parameters and the cost function of those parameters
g, Cost= gradient_descent(X,y,theta,alpha,iters)

def predict(theta, X):  
    return (X * theta.T)

y_pred = predict(g, X)

# Plotting the values
x=np.linspace(data["Population"].min(),data["Population"].max(),100)
f= g[0,0] + g[0,1]*x
plt.scatter(data["Population"], data["Profit"], label="Training data")
plt.plot(x,f, color="r", label="Prediction")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.show()



