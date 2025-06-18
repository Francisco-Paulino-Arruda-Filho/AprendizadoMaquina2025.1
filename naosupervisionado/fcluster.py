from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

mergings = linkage(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
    method='complete'
)

labels = fcluster(mergings, t=1.5, criterion='distance')
print(labels)

import pandas as pd
samples = pd.read_csv('https://raw.githubusercontent.com/joaopaulomoraes/naosupervisionado/main/datasets/iris.csv').drop(columns=['species'])   
pairs = pd.DataFrame(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
    columns=samples.columns
)

print(pairs.sort_values(by=samples.columns[0], ascending=False))