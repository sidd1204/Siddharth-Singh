import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data= pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','species'])


data["species"].value_counts()
data.head()
data.groupby("species").describe().transpose().loc["petal length"]
data.isnull().sum()


data.boxplot(by="species",figsize=(15,15))
data.info()
data.columns

X=data.drop("species",axis=1)
X.head()
Y=data["species"]
Y.head()
len(Y)

from sklearn.preprocessing import LabelEncoder
labelencoder_Y= LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
Y

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size= 0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(X_train,y_train)
Y_pred=classifier.predict(X_test)

Y_train= classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
CM= confusion_matrix(Y_pred,y_test)
CM

