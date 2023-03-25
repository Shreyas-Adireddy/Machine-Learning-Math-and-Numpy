import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self,  cluster_count=3, iters=10):
        self.cluster_count = cluster_count
        self.iters = iters
        self.X = None
        self.centers = None
        self.distances = None

    def fit(self, X):
        # Choose 3 random centers and calculate distance from centers
        self.X = X
        centers = X[np.random.randint(0, X.shape[0], size=self.cluster_count)]
        distances = np.argmin(cdist(X, centers), axis=1)

        for _ in range(self.iters):
            prev_dist = distances.copy()
            # Calculate new centers based on distances
            centers = [np.mean(X[distances==i], axis=0) for i in range(self.cluster_count)]
            centers = np.array(centers)
            distances = np.argmin(cdist(X, centers), axis=1)

            if all(distances==prev_dist):
                break

        self.distances, self.centers = distances, centers
        return centers, distances

    def plot(self):
        # Plots points and centers
        ax = plt.gca()
        ax.scatter(self.X[:,0], self.X[:,1], c=self.distances, s=40, cmap="autumn")
        ax.scatter(self.centers[:,0], self.centers[:,1], cmap="plasma")


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    KM = KMeans(clusters)
    y_pred = KM.fit(X)
    plt.figure(figsize=(8, 6))
    KM.plot()
    plt.show()

