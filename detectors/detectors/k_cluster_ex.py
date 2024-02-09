import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)

print(features[:5])
true_labels[:5]

scaler = StandardScaler()
scaled_features = features

scaled_features[:5]

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=20,
    random_state=42
)

kmeans.fit(scaled_features)

plt.scatter(scaled_features[:,0], scaled_features[:,1], s=2, color='blue')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=50, color='red')
plt.show()

