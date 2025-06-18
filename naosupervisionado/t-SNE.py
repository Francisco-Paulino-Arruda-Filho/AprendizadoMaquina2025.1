from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    random_state=42
)

samples = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
]

model.fit(samples)
transformed_samples = model.fit_transform(samples)

xs = transformed_samples[:, 0]
ys = transformed_samples[:, 1]

plt.scatter(xs, ys)
plt.title('t-SNE Visualization')
plt.xlabel('Component 1')   
plt.ylabel('Component 2')
plt.show()