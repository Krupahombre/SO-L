import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import pairwise_distances, DistanceMetric

X, _ = make_classification(
    n_samples=10000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=1410
)


def k_means(X, n_clusters):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
    distance_metric = DistanceMetric.get_metric('euclidean')

    while True:
        distances = distance_metric.pairwise(X, centroids)
        closest_centroid = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[closest_centroid == i].mean(axis=0) for i in range(n_clusters)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, closest_centroid


n_clusters = 3
centroids, labels = k_means(X, n_clusters)

cmap = plt.colormaps["tab10"]
colors = [cmap(i / n_clusters) for i in range(n_clusters)]

for c in range(n_clusters):
    plt.scatter(X[labels == c, 0], X[labels == c, 1], c=[colors[c]], label=f'Cluster {c + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', label='Centroids', marker='x')
plt.legend()
plt.show()
