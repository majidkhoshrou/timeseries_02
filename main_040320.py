# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:00:53 2020

@author: MajidKhoshrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.style.available
plt.style.use('fivethirtyeight')

from scipy import stats
from sklearn.preprocessing import minmax_scale

from scipy.stats import zscore

def getProb(x):
    y = np.diff(x)
    return 1-(stats.norm.sf(abs(zscore(y)[-1]))*2) #twosided

low_quantile = .05
def get_low_quantile(x):
  y = x.copy()
  return np.quantile(y, low_quantile )

def TrendChangeProb(y):
#    print(x)
    x = y.copy()
    x += np.abs(min(x))
#    x -= np.mean(x)
    a = np.array([ np.trapz((np.random.permutation(x))) for i in range(1000)])
    b= np.trapz((x))
    xx= np.append(a,b)
#    print('xx: ', x)
    return 1-(stats.norm.sf(abs(zscore(xx)[-1])))
    

###############################################################################
# read keyhole data
df = pd.read_csv('data/keyhole_target_emotion_240220.csv', index_col='Created_timestamp', parse_dates=['Created_timestamp'])
df.sort_index( inplace=True)
assert all(df.sort_index().index == df.index)
df.drop( columns=['Date','mood','movement'] , inplace=True )
df.rename( columns={'mood_double':'mood', 'movement_double':'movement'}, inplace=True )

df_4jXEOd = df.loc[ (df["hash"]=='4jXEOd') & (df["platforms"]=='all') ,:]

df_4jXEOd  = df_4jXEOd.resample('D').mean()
df_4jXEOd.rolling(window=7,min_periods=3).apply(getProb, raw=True)

pd.concat([df_4jXEOd.diff(), df.groupby(['hash','platforms']).resample('D').mean().loc["4jXEOd","all"].diff()], axis=1)




df22 = df.groupby(['hash','platforms']).resample('D').mean().interpolate(method='linear', order=5).reset_index()
df22.set_index( 'Created_timestamp', inplace=True )

grouped = df22.groupby(['hash','platforms'])

pd.concat([df22.reset_index()[['mood']] , grouped.rolling(5, min_periods=3).apply(get_low_quantile).reset_index().head(60)], axis=1).head(40)

df22.reset_index()['mood'].head(40)



grouped.


low_quantile = .05
def get_low_quantile(x,low_quantile):
  y = x.copy()
  return np.quantile(y, low_quantile )

grouped.rolling(5, min_periods=3).apply(lambda row: get_low_quantile(row, low_quantile)).reset_index().head(60)






df1 = df.groupby(['hash','platforms']).resample('D').mean().fillna(method='ffill').fillna(0).rolling(window=7,min_periods=3).apply(getProb, raw=True)
# anomaly
df1[['mood_flag','movement_flag']] = df1>.98
df2 = df1.reset_index()
df2.loc[df2.mood_flag,:].shape

# trend setection
window_len = 30
min_periods = 3
low_percentile = .05
high_percentile = .95
df1 =  df.groupby(['hash','platforms']).resample('D').mean().fillna(method='ffill')['mood']
downward_trends = ( df1 < df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=low_percentile) ).astype(int).diff() 
downward_trends = df1.loc[downward_trends.values==1]



