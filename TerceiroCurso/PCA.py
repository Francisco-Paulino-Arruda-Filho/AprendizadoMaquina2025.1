from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris.csv')
df_scaled = df.drop(columns=['species'])

pca = PCA()

df_pca = pca.fit_transform(df_scaled)

print(df_pca)

print(pca.explained_variance_ratio_)