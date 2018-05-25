# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the train data
train = pd.read_csv("train.csv")


# Analysing the data
train.head()
train.dtypes
len(train.columns)
train.columns

# Analysing the overall chance of survival for a Titanic passenger.
train["Survived"].mean()
train["Pclass"].value_counts()
train.groupby("Pclass").describe().transpose().loc["Survived"]
train.groupby("Pclass").describe()

train.groupby(["Pclass","Sex",]).describe().transpose().loc["Survived"]


# Removing the columns which doesn't play important role in the survival
train.drop(["Name","Cabin","PassengerId","Ticket"],axis = 1,inplace = True)
train.describe()

# Handling Missing Values In Train Data using pandas
train.isnull().sum()
train.Age.fillna(np.mean(train.Age),inplace= True)
train.Embarked.fillna("Not_Specified",inplace= True)

# Encoding Categorical Data using pandas
TRAIN=pd.get_dummies(train, columns=["Sex", "Embarked","Pclass"], prefix=["Sex", "Embarked","Pclass"])
TRAIN.head(10)
TRAIN.describe()
len(TRAIN.columns)
Y_TRAIN= TRAIN.Survived
X_TRAIN= TRAIN.drop("Survived", axis= 1)
X_TRAIN.ndim
Y_TRAIN.ndim

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_TRAIN[["Age","Fare"]] = sc.fit_transform(X_TRAIN[["Age","Fare"]])
X_TRAIN.head(5)
len(X_TRAIN.columns)



# Reading the test data
test= pd.read_csv("test.csv")
test.drop(["Name","Cabin","PassengerId","Ticket"],axis = 1,inplace = True)
test.head(20)

# Handling missing values in test data
test.isnull().sum()
test.Age.fillna(np.mean(test.Age),inplace = True)
test.Fare.fillna(np.mean(test.Fare),inplace = True)

# Encoding Categorical Data
TEST= pd.get_dummies(test, columns=["Sex","Embarked","Pclass"],prefix = ["Sex","Embarked","Pclass"])
TEST.head(10)
X_TEST= TEST
len(X_TEST.columns)
X_TRAIN.drop("Embarked_Not_Specified",axis= 1,inplace= True)
len(X_TRAIN.columns)
X_TEST.head()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_test= StandardScaler()
X_TEST[["Age","Fare"]]= sc_test.fit_transform(X_TEST[["Age","Fare"]])


# Applying the Logistic Regression Classification Model
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()
# Fitting our model on the train dataset
LR.fit(X_TRAIN, Y_TRAIN)
# Predicting the test dataset
Y_pred= LR.predict(X_TEST)
Y_pred

# Calculating the parameters of our features used
LR.coef_

# Calculating the Accuracy of our model
LR.score(X_TRAIN, Y_TRAIN)

# Predicting the train dataset
Y_pred_TRAIN= LR.predict(X_TRAIN)
Y_pred_TRAIN

# Creating Confusion Matrix to verify the accuracy we got 
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_TRAIN, Y_pred_TRAIN)


