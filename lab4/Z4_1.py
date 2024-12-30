from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def k_means_parallel(X, n_clusters):
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)] if rank == 0 else None
    centroids = comm.bcast(centroids, root=0)

    final_labels = None
    if rank == 0:
        final_labels = np.zeros(X.shape[0], dtype=int)

    while True:
        data_chunks = np.array_split(X, size) if rank == 0 else None
        local_X = comm.scatter(data_chunks, root=0)
        distance_metric = DistanceMetric.get_metric('euclidean')
        distances = distance_metric.pairwise(local_X, centroids)
        closest_centroid = np.argmin(distances, axis=1)

        local_sums = np.zeros_like(centroids)
        local_counts = np.zeros(n_clusters)

        for i, cluster in enumerate(closest_centroid):
            local_sums[cluster] += local_X[i]
            local_counts[cluster] += 1

        total_sums = np.zeros_like(local_sums)
        total_counts = np.zeros_like(local_counts)

        comm.Reduce(local_sums, total_sums, op=MPI.SUM, root=0)
        comm.Reduce(local_counts, total_counts, op=MPI.SUM, root=0)

        if rank == 0:
            new_centroids = np.array([
                total_sums[i] / total_counts[i] if total_counts[i] > 0 else centroids[i]
                for i in range(n_clusters)
            ])

            if np.allclose(centroids, new_centroids):
                all_closest = comm.gather(closest_centroid, root=0)
                if all_closest:
                    final_labels = np.concatenate(all_closest)
                break

            centroids = new_centroids

        centroids = comm.bcast(centroids, root=0)

    return centroids if rank == 0 else None, final_labels if rank == 0 else None


n_clusters = 3

if rank == 0:
    X, _ = make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=1410
    )
else:
    X = None

centroids, labels = k_means_parallel(X, n_clusters)

if rank == 0:
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i / n_clusters) for i in range(n_clusters)]

    for c in range(n_clusters):
        plt.scatter(X[labels == c, 0], X[labels == c, 1], c=[colors[c]], label=f'Cluster {c + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', label='Centroids', marker='x')
    plt.legend()
    plt.show()
