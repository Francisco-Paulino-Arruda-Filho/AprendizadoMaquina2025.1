from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

pca = PCA(
    n_components=2,
    svd_solver='full',
    random_state=42
)

iris = load_iris()
X = iris.data
y = iris.target
pca.fit(X)

X_transformed = pca.transform(X)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Species')
plt.show()