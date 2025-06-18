import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

samples = [
    [1.0, 2.0, 3.0],
    [1.5, 1.8, 2.5],
    [5.0, 8.0, 7.0],
    [8.0, 8.0, 9.0],
    [1.0, 0.6, 1.2],
    [9.0, 11.0, 10.5]
]

features = ['feature1', 'feature2', 'feature3']

pca = PCA()
pca.fit(samples)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()