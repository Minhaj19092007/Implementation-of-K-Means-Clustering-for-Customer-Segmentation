# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Import data, inspect it, and handle missing values if any.
2. Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.
3. Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.
4. Assign cluster labels to each data point.
5. Plot data points in a scatter plot, color-coded by cluster assignments for interpretation.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Minhaj Ahamed J 
RegisterNumber:  212225230171
*/


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation using K-Means")

plt.show()

```

## Output:
<img width="1389" height="574" alt="image" src="https://github.com/user-attachments/assets/c1896315-88ff-41e3-8edb-e3ff214508b2" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
