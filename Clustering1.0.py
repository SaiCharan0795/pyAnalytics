# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:37:49 2021

@author: Hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns
data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50, 57,59,52,65, 47,49,48,35,33,44,45, 38,43,51,46],'y': [79,51,53, 78,59,74,73,57, 69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14, 12,20,5,29, 27,8,7]  }
data  
df = pd.DataFrame(data, columns = ['x','y'])
df.head()
df.mean()
df.max()
df.min()
df.shape
print(df)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(df)
dir(kmeans)
kmeans.n_clusters
centroids = kmeans.cluster_centers_
print(centroids)
kmeans.labels_
df
plt.scatter(df['x'],df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='D')
plt.show()

#four clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4).fit(df)
dir(kmeans)
kmeans.n_clusters
centroids = kmeans.cluster_centers_
print(centroids)
kmeans.labels_
df
plt.scatter(df['x'],df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='D')
plt.show()

#mtcarsData
from pydataset import data
mtcars = data('mtcars')
mtcarsData = mtcars.copy()
id(mtcarsData)
mtcarsData

# 3 clusters from kmeans
kmeans = KMeans(n_clusters=3).fit(mtcarsData)
kmeans.n_clusters
kmeans.inertia_
centroids = kmeans.cluster_centers_
print(centroids)
kmeans.labels_
mtcarsData.groupby(kmeans.labels_).aggregate({'mpg':[np.mean, 'count']})
mtcarsData.min(),mtcarsData.max()

#need for scaling : height & weight are in different scales
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mtcarsScaledData = scaler.fit_transform(mtcarsData)
mtcarsScaledData[:5] # values between -3 to +3
