
# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


## DELIVERABLE 1

# Load the crypto_data.csv dataset.
file_path = "Resources/crypto_data.csv"
crypto_df = pd.read_csv(file_path, index_col=0)
crypto_df.head(10)


#check shape beforehand
crypto_df.shape


# Keep all the cryptocurrencies that are being traded.
crypto_df = crypto_df[crypto_df.IsTrading != 0]
print(crypto_df.shape)
crypto_df.head(10)



# Keep all the cryptocurrencies that have a working algorithm.
crypto_df = crypto_df[crypto_df.Algorithm.isna() == False]
print(crypto_df.shape)
crypto_df.head(10)



# Remove the "IsTrading" column. 
crypto_df = crypto_df.drop(columns="IsTrading")
print(crypto_df.shape)
crypto_df.head(10)


# Remove rows that have at least 1 null value.
crypto_df = crypto_df[~(crypto_df.isna() == True).any(axis=1)]
print(crypto_df.shape)
crypto_df.head(10)


# Keep the rows where coins are mined.
crypto_df = crypto_df[~(crypto_df.TotalCoinsMined <= 0)]
print(crypto_df.shape)
crypto_df.head(10)


# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_name_df = pd.DataFrame(crypto_df.CoinName)
print(crypto_name_df.shape)
crypto_name_df.head()



# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_df = crypto_df.drop(columns="CoinName")
print(crypto_df.shape)
crypto_df.head(10)

# Use get_dummies() to create variables for text features.
X = pd.get_dummies(crypto_df, columns=["Algorithm", "ProofType"])
print(X.shape)
X.head(10)

# Standardize the data with StandardScaler().
scaler = StandardScaler()
crypto_scaled = scaler.fit_transform(X)
crypto_scaled[0:5]


## Deliverable 2: Reducing Data Dimensions Using PCA

# Using PCA to reduce dimension to three principal components.
pca = PCA(n_components=3)
crypto_pca = pca.fit_transform(crypto_scaled)
crypto_pca


# Create a DataFrame with the three principal components.
pcs_df = pd.DataFrame(crypto_pca, columns=["PC 1", "PC 2", "PC 3"], index=crypto_name_df.index)
print(pcs_df.shape)
pcs_df.head(10)


### Deliverable 3: Clustering Crytocurrencies Using K-Means
# Create an elbow curve to find the best value for K.

# Initialize empty inertia list
inertia = []
# Initialize k range
k = list(range(1,11))
# Looping through k list
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(pcs_df)
    inertia.append(km.inertia_)

# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
elbow_df = pd.DataFrame(elbow_data)
elbow_df.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)


##Running K-Means with k4

# Initialize the K-Means model.
km_model = KMeans(n_clusters=4, random_state=0)

# Fit the model
km_model.fit(pcs_df)

# Predict clusters
predictions = km_model.predict(pcs_df)
predictions


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = pd.concat([crypto_df, pcs_df], axis=1, join='inner')

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df["CoinName"] = crypto_name_df.CoinName

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df["Class"] = km_model.labels_

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)



### Deliverable 4: Visualizing Cryptocurrencies Results¶

# Creating a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(clustered_df, x="PC 1", y="PC 2", z="PC 3", color="Class", symbol="Class", width=800, 
                    hover_name="CoinName", hover_data=["Algorithm"])
fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# Create a table with tradable cryptocurrencies.
tradable_crypto_table = clustered_df.hvplot.table(columns=['CoinName', 'Algorithm', 'ProofType', 'TotalCoinSupply', 
                                                  'TotalCoinsMined', 'Class'], sortable=True, selectable=True)
tradable_crypto_table



# Print the total number of tradable cryptocurrencies.
print(f'There are {len(clustered_df)} tradable cryptocurrencies.')


# Scaling data to create the scatter plot with tradable cryptocurrencies.
X_cluster = clustered_df[['TotalCoinSupply', 'TotalCoinsMined']].copy()
X_cluster_scaled = MinMaxScaler().fit_transform(X_cluster)
X_cluster_scaled


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
plot_df = pd.DataFrame(X_cluster_scaled, columns=['TotalCoinSupply', 'TotalCoinsMined'], index=clustered_df.index)

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
plot_df['CoinName'] = clustered_df.CoinName

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
plot_df['Class'] = clustered_df.Class

plot_df.head(10)



# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
plot_df.hvplot.scatter(x='TotalCoinsMined', y='TotalCoinSupply', hover_cols='CoinName', by='Class')


# PCA with 2 components
# Using PCA to reduce dimension to two principal components.
pca_two = PCA(n_components=2)
crypto_pca_two = pca_two.fit_transform(crypto_scaled)
crypto_pca_two


# Create a DataFrame with the two principal components.
two_pcs_df = pd.DataFrame(crypto_pca_two, columns=["PC 1", "PC 2"], index=crypto_name_df.index)
print(two_pcs_df.shape)
two_pcs_df.head(10)

# Initialize the K-Means model.
km_model_two = KMeans(n_clusters=4, random_state=0)

# Fit the model
km_model_two.fit(two_pcs_df)

# Predict clusters
predictions = km_model_two.predict(two_pcs_df)
predictions



# Create a new DataFrame including predicted clusters and cryptocurrencies features.
two_pcs_clustered_df = pd.concat([crypto_df, two_pcs_df], axis=1, join='inner')

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
two_pcs_clustered_df["CoinName"] = crypto_name_df.CoinName

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
two_pcs_clustered_df["Class"] = km_model_two.labels_

# Print the shape of the clustered_df
print(two_pcs_clustered_df.shape)
two_pcs_clustered_df.head(10)


# Create a hvplot.scatter plot using x="PC 1" and y="PC 2".
two_pcs_clustered_df.hvplot.scatter(x='PC 1', y='PC 2', hover_cols='CoinName', by='Class')

