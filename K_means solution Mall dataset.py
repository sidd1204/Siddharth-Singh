import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")

data.head(10)
    
X = data.iloc[:,3:5].values

# Using Elbow method to find no. of clusters

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("The Elbow Method")
plt.show()

# Applying K-means to the dataset
kmeans= KMeans(n_clusters=5)
y_kmeans= kmeans.fit_predict(X)
Centroids = kmeans.cluster_centers_


# Visualizing the Clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1], s=100,c="red", label="1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1], s=100,c="blue", label="2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1], s=100,c="green", label="3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1], s=100,c="magenta", label="4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1], s=100,c="cyan", label="5")
plt.scatter(Centroids[:,0],Centroids[:,1], s=100,c="yellow", label="Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Scores")
plt.legend()
plt.show()