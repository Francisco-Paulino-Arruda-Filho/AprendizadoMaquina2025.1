import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df['target'] = load_iris().target
