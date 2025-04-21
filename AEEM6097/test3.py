import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from pyclustertend import compute_ivat_ordered_dissimilarity_matrix
from scipy.cluster.vq import kmeans
from scipy.io import loadmat

# Load the Iris2D.mat object
data = loadmat('Iris2D.mat')

# Extract the data from the loaded object
iris_data = data['X'].astype(np.float64)

# Use the IVAT technique to identify how many clusters are present
# Compute the ordered dissimilarity matrix
ordered_dissimilarity_matrix = compute_ivat_ordered_dissimilarity_matrix(iris_data)
# Plot the ordered dissimilarity matrix
plt.imshow(ordered_dissimilarity_matrix)
plt.colorbar()
plt.title('Ordered Dissimilarity Matrix')
plt.show()

# Perform knee-finding with k-means clustering from k=1 to 10 and plot the error
# This is a simple elbow method to find the optimal number of clusters
error_history = []
for k in range(1, 11):
    centroids, distortion = kmeans(iris_data, k)
    error_history.append(distortion)

plt.figure()
plt.plot(range(1, 11), error_history, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('K-means Elbow Method')

# There are two clearly large clusters, several subclusters, but lets ignore them.
km = kmeans(iris_data, 2)
# Print the cluster centers
print("K-means Cluster centers:\n", km[0])
# Perform the fuzzy c-means clustering
fcm = FCM(n_clusters=2, m=2, max_iter=100, error=0.0005)
error_history = fcm.fit(iris_data)

# Plot the error history
plt.figure()
plt.plot(error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Fuzzy C-Means Error History')

fcm_centers = fcm.centers
print("Fuzzy C-Means Cluster Centers:\n", fcm_centers)
# Print the membership degree of each point to each cluster
membership_degree = fcm.soft_predict(iris_data)

# Plot the membership degree for the first two clusters
plt.figure()
plt.plot(membership_degree)
plt.title('Membership Degree for First Two Clusters')
plt.xlabel('Membership Degree')
plt.ylabel('Data Points')
plt.legend(['Cluster 1', 'Cluster 2'])
plt.show()

# Plot the data points and cluster data
plt.figure()
ax = plt.scatter(iris_data[:, 0], iris_data[:, 1])
plt.scatter(km[0][:, 0], km[0][:, 1], c='red', marker='x', label='K-means Centroids')
plt.scatter(fcm_centers[:, 0], fcm_centers[:, 1], c='purple', marker='x', label='Fuzzy Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()