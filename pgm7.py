import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Your data
X = np.array([[1., 1.],
              [1.5, 2.],
              [3., 4.],
              [5., 7.],
              [3.5, 5.],
              [4.5, 5.],
              [3.5, 4.5]])

# KMeans Clustering
kmeans = KMeans(n_clusters=2)
labels_kmeans = kmeans.fit_predict(X)
print("Labels for KMeans:", labels_kmeans)

# Plot using KMeans Algorithm
print('Graph using KMeans Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red')
plt.show()

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2)
labels_gmm = gmm.fit_predict(X)
print("Labels for GMM: ", labels_gmm)

# Plot using GMM Algorithm
print('Graph using EM Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels_gmm)
plt.show()
