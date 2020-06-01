# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:58:12 2020

@author: MajidKhoshrou
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import zscore
from scipy import stats

#np.random.seed(8)
#ts1 = np.random.rand(.2,.4, size=n1)
#ts1 = 20-np.array([1,2.4,3.5,4,5,6,5.5,6.3, 6.2,5.1])
#n1=len(ts1)

n1 = 100
np.random.seed(44)
#ts1 = np.random.randn(n1)
mu=0
sigma=1
ts1 = np.cumsum(sigma*np.random.randn(n1)+mu)
ts1[0]=10
idx1 = pd.date_range('2019-10-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
#df1=df1.rolling(window=10, min_periods=2).mean().fillna(method='bfill')
df1=df1.rolling(window=10, min_periods=2).mean().fillna(method='bfill')

#df1.plot(marker='o', linestyle='-', linewidth=2, figsize=(10, 8))

#x =  df1.loc['9 Dec 2019':'23 Dec 2019'].ts.values

def TrendChangeProb(y):
#    print(x)
    x = y.copy()
    x += np.abs(min(x))
#    print('shifted: ', x)
    a = np.array([ np.trapz((np.random.permutation(x))) for i in range(1000)])
    b= np.trapz((x))
    xx= np.append(a,b)
#    print('xx: ', x)
    return 1-(stats.norm.sf(abs(zscore(xx)[-1])))
    
    
    
idx = df1.rolling(14).apply(TrendChangeProb, raw=True)>.95

df2 = df1.loc[idx.ts.values,:]
ax2 = df1.plot(figsize=(20,15), marker='o', linewidth=3)
for d in df2.index:
    ax2.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)
  
    

ffff

ax2 = df1.diff().plot(figsize=(20,15), marker='o', linewidth=3)


def getProb(x):
    return 1-(stats.norm.sf(abs(zscore(x)[-1]))*2) #twosided

getProb(x)







df2 = pd.DataFrame({'ts':np.random.permutation(ts1)}, index=idx1)
df2.plot()

df2.cumsum().mean()

np.diff(ts1)


#df1.diff().plot(figsize=(10, 8), linewidth=1, kind='hist')
df1.diff().plot(figsize=(10, 8), linewidth=1, marker='o')

df1.diff().mean()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

from scipy.signal import savgol_filter

df3 = pd.DataFrame({"ts": savgol_filter(ts1, window_length=3,polyorder=1, deriv=0) }, index=idx1)
pd.concat([df1,df3], axis=1).plot(marker='^')


trapz1 = np.trapz(np.diff(ts1))
trapz2 = np.trapz(np.diff(np.random.permutation(ts1)))
print('{}-->{}'.format( trapz1, trapz2))

np.min(ts1)
trapz1 = np.trapz((ts1))
trapz2 = np.trapz((np.random.permutation(ts1)))
print('{}-->{}'.format( trapz1, trapz2))

np.mean(np.array([np.trapz((np.random.permutation(ts1))) for i in range(1000)]))
np.std(np.array([np.trapz((np.random.permutation(ts1))) for i in range(1000)]))









