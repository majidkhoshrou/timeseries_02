# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:58:41 2020

@author: MajidKhoshrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import seaborn as sns
from sklearn.linear_model  import LinearRegression

plt.style.available
plt.style.use('fivethirtyeight')

###############################################################################
# read keyhole data
df = pd.read_csv('data/keyhole_target_emotion_240220.csv', index_col='Created_timestamp', parse_dates=['Created_timestamp'])
df.sort_index( inplace=True)
assert all(df.sort_index().index == df.index)

df1 = df.loc[(df.search_terms=='philips') & (df.platforms=='all')].loc[:,['mood','movement']]
df1 = df1.resample('D').mean().fillna(method='ffill')
df1.plot(subplots=True, layout = (2,1), linewidth=2, figsize=(10, 8), marker='o')

###############################################################################
# generate random data to design a model
n1 = 100
np.random.seed(8)
ts1 = np.random.randn(n1)
ts1[0]=10
ts1 = np.cumsum(ts1)
idx1 = pd.date_range('2018-01-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(10).mean().fillna(method='bfill')
df1.iloc[25]=17
df1.iloc[40:41]=4

df1.iloc[75:85]=3
df1.iloc[85:]=12

df1.ts = df1.ts + np.random.randn(n1)/20
df1.plot(figsize=(10, 8), marker='o', linewidth=1)
###############################################################################
# develope a model

windowLength = 10


df1 = df1.replace([np.inf, -np.inf], np.nan)
df1.fillna(method='bfill', inplace=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from scipy import stats


n_tree=windowLength
rf = RandomForestRegressor(n_estimators=n_tree, max_depth=2, random_state=0, criterion='mae', max_features=None)
#rf = GradientBoostingRegressor(loss='huber',n_estimators=n_tree)

startDate = pd.to_datetime('January 3, 2018')
endDate = pd.to_datetime('April 10, 2018')

d1 = startDate - datetime.timedelta(days=1)
df2 = df1.copy()

df2['zScore']=0
for d in pd.date_range(startDate, endDate):
    
    d1 = d - datetime.timedelta(windowLength+1)
    
    X = df2.loc[d1:d,'ts']
    
    z = np.abs(stats.zscore(X))
    df2.loc[d,'zScore'] = z[-1]

print(df2.loc[df2.zScore>3,:])
    
  

    
    
    
    
    
    
    
    
    
    