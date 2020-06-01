# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:50:04 2020

@author: MajidKhoshrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import zscore
from scipy import stats

import statsmodels.api as sm


#np.random.seed(8)
#ts1 = np.random.rand(.2,.4, size=n1)
#ts1 = 20-np.array([1,2.4,3.5,4,5,6,5.5,6.3, 6.2,5.1])
#n1=len(ts1)

n1 = 100
np.random.seed(33)
#ts1 = np.random.randn(n1)

mu=0
sigma=1
ts1 = np.cumsum(sigma*np.random.randn(n1)+mu)

idx1 = pd.date_range('2019-10-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(window=7, min_periods=1).mean().fillna(method='bfill')
#df1=df1.round()



#df1=df1.rolling(window=10, min_periods=2).mean().fillna(method='bfill')
#df1.plot(marker='o', linestyle='-', linewidth=2, figsize=(10, 8))

#x =  df1.loc['9 Dec 2019':'23 Dec 2019'].ts.values

def getProb(x):
    
    return 1-(stats.norm.sf(abs(zscore(x)[-1]))*2) #twosided


def TrendChangeProb(y):
        
    x = y.copy()
    x += np.abs(min(x))
#    x -= np.mean(x)
    a = np.array([ np.trapz((np.random.permutation(x))) for i in range(1000)])
    b= np.trapz((x))
    xx= np.append(a,b)
    
#    print('xx: ', x)
    return 1-(stats.norm.sf(abs(zscore(xx)[-1])))

thresh = .95


idx = df1.rolling(window=14, min_periods=1).apply(TrendChangeProb, raw=True)>.95

#df1['trend_change_prob']  = df1.ts.rolling(window=7, min_periods=1).apply(TrendChangeProb, raw=True)
#df1['trend_change_prob2']  = df1.trend_change_prob.rolling(window=7, min_periods=1).apply(TrendChangeProb2, raw=True)
#
#
#df1.plot(figsize=(20,15), marker='o', linewidth=3)
#
#idx = df1['trend_change_prob2']==1

#idx2 = idx.ts.astype(int)

#idx2.resample('W').sum()

#idx = df1.diff().rolling(window=7, min_periods=1).std().apply(getProb, raw=True)>.9


df2 = df1.loc[idx.ts.values,:]
ax2 = df1.plot(figsize=(20,15), marker='o', linewidth=3)
for d in df2.index:
    ax2.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)



dd
#########################################################################################





def myFunc(s):
    return np.trapz(s)
a = df1.rolling(window=5, min_periods=1).apply(myFunc, raw=True)
#pd.concat([df1, a], axis=1).plot(figsize=(20,15), marker='o', linewidth=3)

a.pct_change().plot()

df1.plot(figsize=(20,15), marker='o', linewidth=3)

df1.diff().rolling(window=5, min_periods=1).std().plot(figsize=(20,15), marker='o', linewidth=3)

x = df1.ts.values
import pywt
(cA, cD) = pywt.dwt(x, 'db1')

plt.plot(cD)

#df1.rolling(window=7, min_periods=3).apply(np.trapz).pct_change().plot()

decomposed_ts = sm.tsa.seasonal_decompose(df1["ts"],freq=14)

pd.concat([df1, decomposed_ts.trend], axis=1).plot()


window_len = 30
min_periods = 3
low_percentile = .05
high_percentile = .95

df2 = pd.concat([df1, df1.rolling(window=window_len,min_periods=min_periods).min(),\
                 df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=.05),\
                 df1.rolling(window=window_len,min_periods=min_periods).median(),\
                 df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=.95),\
                 df1.rolling(window=window_len,min_periods=min_periods).max(),\
                 df1.expanding().min()], axis=1)
df2.columns = ['ts','min', '5%', 'mean','95%','max','expanding']
df2.plot(figsize=(20,15), marker='o', linewidth=3)

a = df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=low_percentile)
b = (df1 < a).astype(int).diff()
drop_below_low_percentile = ( df1 < df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=low_percentile) ).astype(int).diff()
drop_below_low_percentile.plot(figsize=(20,15), marker='o', linewidth=3);

downward_trends = ( df1 < df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=low_percentile) ).astype(int).diff() 
downward_trends = df1.loc[downward_trends.ts.values==1]
pd.concat([df1, a, b], axis=1).plot(figsize=(20,15), marker='o', linewidth=3)

c = df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=high_percentile)
d = (df1 > c).astype(int).diff()
pd.concat([df1, c, d], axis=1).plot(figsize=(20,15), marker='o', linewidth=3)

upward_trends = (df1 > df1.rolling(window=window_len,min_periods=min_periods).quantile(quantile=high_percentile)).astype(int).diff()
upward_trends = df1.loc[upward_trends.ts.values==1,:]


e = df1.rolling(window=window_len,min_periods=min_periods).median()
f = (df1 < e).astype(int).diff()
pd.concat([df1, e, f], axis=1).plot(figsize=(20,15), marker='o', linewidth=3)

#(df1 >= df1.expanding().quantile(quantile=.5)).replace({True:1, False:-1}).astype(int).pct_change().plot(figsize=(20,15), marker='o', linewidth=3);








pd.concat([df1, df1.shift(), df1.pct_change(), (df1.diff()/df1.shift())], axis=1)



a = df1< df1.rolling(window=3,min_periods=1).median()
a=a.astype(int)
pd.concat([df1, df1.rolling(window=3,min_periods=1).median(), a], axis=1).head(30)


    























