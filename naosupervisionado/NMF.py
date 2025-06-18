from sklearn.decomposition import NMF
import numpy as np

model = NMF(
    n_components=2,
    init='random',
    random_state=42
)

samples = [
    [1.0, 2.0, 3.0],
    [1.5, 1.8, 2.5],
    [5.0, 8.0, 7.0],
    [8.0, 8.0, 9.0],
    [1.0, 0.6, 1.2],
    [9.0, 11.0, 10.5]
]

model.fit(samples)
transformed_samples = model.transform(samples)  
print("Components:\n", model.components_)
nmf_features = model.fit_transform(samples)
print("NMF Features:\n", nmf_features)
