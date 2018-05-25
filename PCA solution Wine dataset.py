# Importing the required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the Wine dataset
data= pd.read_csv("Wine.csv")
X= data.iloc[:,:-1]
Y= data.iloc[:,-1]

# Spliting the data in train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# Applying Feature Scalaing
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Applying PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components =  2)
X_train = pca.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns = ["Z1","Z2"])
X_test = pca.transform(X_test)
X_test = pd.DataFrame(X_test, columns = ["Z1","Z2"])
explained_variance = pca.explained_variance_ratio_ 
ev= pca.explained_variance_

# Fitting and transforming using LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred= classifier.predict(X_test)

# Finding Accuracy of our model
classifier.score(X_test,Y_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm


