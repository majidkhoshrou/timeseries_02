# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:25:41 2020

@author: MajidKhoshrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import seaborn as sns
from sklearn.linear_model  import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.style.available
plt.style.use('fivethirtyeight')

from scipy import stats
from sklearn.preprocessing import minmax_scale

from scipy.stats import zscore

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
#n1 = 100
#np.random.seed(404)
#ts1 = np.random.randn(n1)
##ts1 = np.random.beta(.2,.4, size=n1)
#ts1[0]=10
#mu=-0.5
#sigma=1
#ts1 = np.cumsum(sigma*ts1+mu)
#idx1 = pd.date_range('2019-10-01', periods=n1, freq='D')
#df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
#df1=df1.rolling(3).mean().fillna(method='bfill')
#
#df1.ts = (df1.ts + np.random.randn(n1)/20)
#
#df1.iloc[10:20] /= 2.0
#df1.iloc[35:36] -= 5
#
#df1.iloc[50:80] += 1.2
#df1.iloc[80:] -= 1



###############################################################################
# develope a model

windowLength = 4

df1 = df1.replace([np.inf, -np.inf], np.nan)
df1.fillna(method='bfill', inplace=True)
df100 = df1.copy()
df100['y_hat']=np.nan

vals = df1.values
reg = Lasso(alpha=1)
#reg = LinearRegression()
change_flag = np.zeros(len(vals))

for i in range(windowLength, len(vals)):
    
    y = vals[i-windowLength:i].reshape(-1, 1)
    X = np.array(list(range(len(y)))).reshape(-1,1)
    
    y_train = y[:-1]
    X_train = X[:-1]
    
    y_test = y[-1]
    X_test = X[-1,:].reshape(1, -1)
    
    reg = reg.fit(X_train, y_train)

    reg.coef_
    reg.intercept_

    y_hat = reg.predict(X_test)
    df100.iloc[i,-1] = y_hat
    
    res_train = y_train - reg.predict(X_train).reshape(-1, 1)
    res_test = y_test - y_hat
    
    flagIdx = res_test> 2*np.std(y_train)
    change_flag[i] = res_test> 3*np.std(res_train)
    
#    plt.plot(X ,y, 'r-',X_test, y_hat, '*',linewidth=2)

change_flag = change_flag.astype(bool)
np.sum(change_flag)


##########################################################

def getZscore(x):
    print(x)
    return zscore(x)[0]

def getProb(x):
    y = np.diff(x)
    return 1-(stats.norm.sf(abs(zscore(y)[-1]))*2) #twosided

def getProb1(x):
    """
    One sided, suitable for abs of diff.
    """
    return (stats.norm.sf(abs(zscore(x)[-1]))) 

def getProb2(x):
    mu1 = np.nanmean(x[:-1])
    sigma1 = np.nanstd(x[:-1])
    mu2 = np.nanmean(x)
    sigma2 = np.nanstd(x)
    z_score = (mu1-mu2)/(sigma1/np.sqrt(len(x)-1))
    return 1-(stats.norm.sf(abs(z_score))*2)
    
  
#df1.diff().rolling(10,min_periods=3).apply(getZscore, raw=True)

#[v for _,v in df1.itertuples()]
idx = df1.fillna(0).rolling(window=14,min_periods=3).apply(getProb, raw=True) >.95

#idx = df1.diff().abs().rolling(14,min_periods=4).apply(getProb1, raw=True)>.95

df2 = df1.loc[idx.ts.values,:]
ax2 = df1.plot(figsize=(20,15), marker='o', linewidth=3)
for d in df2.index:
    ax2.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)

#################################################################
mm


df1 = df.loc[(df.search_terms=='philips') & (df.platforms=='all')].loc[:,['mood']]
df1 = df1.resample('D').mean().fillna(method='ffill')
#idx = df1.diff().rolling(14, min_periods=4).apply(getZscore, raw=True).abs()>2








idx = df1.diff().rolling(5, min_periods=4).apply(getProb, raw=True)>.98


df2 = df1.loc[idx.mood.values,:]
ax2 = df1.plot(figsize=(20,15), marker='o', linewidth=3)
for d in df2.index:
    ax2.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)

plt.plot(zscore(df1))

def len_window(x):
    return (x[0])

df1['x'] = df1.mood.rolling(window=10, closed='right').apply(len_window)
    
z_scores = (6.9-7.02)/(.84/np.sqrt(202))
stats.norm.sf(z_scores)

from scipy import special
special.ndtr(-z_scores)

import numpy as np
import scipy.special as scsp
def z2p(z):
    """From z-score return p-value."""
    return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))
    
z2p(z_scores)   
    
    
1-(stats.norm.sf((z_scores)))
    
stats.norm.sf(abs(z_scores))
getProb(z_scores)
1-(stats.norm.sf(z_scores)*2)

zscore(np.array([100,3,4,5,7,8,200]))

stats.norm.sf(abs(z_scores))

a=df1.index[4]
b=df1.index[9]
min(a,b)
