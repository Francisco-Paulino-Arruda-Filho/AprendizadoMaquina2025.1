import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

samples = pd.read_csv('https://raw.githubusercontent.com/joaopaulomoraes/naosupervisionado/main/datasets/iris.csv').drop(columns=['species'])

mergings = linkage(
    samples,
    method='complete'
)
plt.figure(figsize=(10, 7))
dendrogram(mergings, samples.index, leaf_rotation=90, leaf_font_size=12)
plt.plot('dendrogram.png')
plt.title('Hierarchical Clustering Dendrogram') 
