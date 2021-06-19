
url ='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv'

import pandas as pd
import numpy as np
df = pd.read_csv(url)
df
df.columns
df.shape
df.head(n=3)
len(df)
df.region.value_counts()
df.region.value_counts().plot(kind='bar')
df.describe()
df['region']=df['region'].astype('category')
df
df.custname.count()
df.custname.value_counts()
df.custname.value_counts().tail()
df
df.groupby('custname').revenue.sum()
df.groupby('custname').revenue.sum().sort_values()
df.groupby('custname').revenue.sum().sort_values(ascending=True).head(5)
df.groupby('custname').aggregate({'revenue':[np.sum,max,min,'size']})
df.groupby('custname').aggregate({'revenue':[np.sum,max,min,'size']}).sort_values(by='sum')
df.groupby('custname')['revenue'].aggregate([np.sum,max,min,'size']).sort_values(by='sum')
df.groupby('partnum').revenue.sum().sort_values()
df.groupby('partnum')['margin'].aggregate([np.sum,max,min,'size']).sort_values(by='sum')
df
df.groupby('partnum').revenue.max().sort_values(ascending=True).head(5)
df.groupby('partnum').size().sort_values(ascending=False).head(5)
df[['revenue','region']].groupby('region').sum().sort_values(by='revenue',ascending=False).plot(kind='bar')
