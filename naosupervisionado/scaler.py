from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans

samples = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
]

scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_scaled = scaler.transform(samples)

kmeans = KMeans(n_clusters=2, random_state=42)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
scaled_samples = pipeline.transform(samples)