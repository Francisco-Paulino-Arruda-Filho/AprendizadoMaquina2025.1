from sklearn.cluster import KMeans

samples = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
]

model = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

new_samples = [
    [0.0, 0.0],
    [10.0, 10.0]
]

model.fit(samples)
print("Cluster centers:", model.cluster_centers_)
print("Labels:", model.labels_)
print("Inertia:", model.inertia_)
print("Number of iterations:", model.n_iter_)
print("Number of clusters:", model.n_clusters_)
print("Number of features:", model.n_features_in_)
print("Prediction for new sample [0.0, 0.0]:", model.predict([[0.0, 0.0]]))
print("Prediction for new sample [10.0, 10.0]:", model.predict([[10.0, 10.0]]))
