from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_flower_dataset():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def kmeans_on_flower_dataset(X_train, n_clusters=3):
    # Create a KMeans model
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # Fit the model to the training data
    model.fit(X_train)

    return model

def main():
    # Load the flower dataset
    X_train, X_test, y_train, y_test = load_flower_dataset()

    # Apply KMeans clustering
    model = kmeans_on_flower_dataset(X_train)

main()