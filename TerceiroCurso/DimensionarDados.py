from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled)
print(df_scaled.var())

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler

# Subset the DataFrame you want to scale 


# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)