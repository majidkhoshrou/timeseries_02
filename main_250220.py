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
np.random.seed(11)
ts1 = np.random.randn(n1)
ts1[0]=10
mu=0
sigma=1
ts1 = np.cumsum(sigma*np.random.randn(n1)+mu)
idx1 = pd.date_range('2019-10-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(10).mean().fillna(method='bfill')
df1.iloc[25]=0
df1.iloc[40:41]=9

df1.iloc[75:]=3 + df1.iloc[75:]
#df1.iloc[85:]=12

df1.ts = (df1.ts + np.random.randn(n1)/10)



###############################################################################
# develope a model

windowLength = 4

df1 = df1.replace([np.inf, -np.inf], np.nan)
df1.fillna(method='bfill', inplace=True)

vals = df1.values
reg = Lasso(alpha=1)
#reg = LinearRegression()
change_flag = np.zeros(len(vals))

for i in range(windowLength, len(vals)):
    
    y = vals[i-windowLength:i].reshape(-1, 1)
    X = np.array(range(len(y))).reshape(-1, 1)
    
    y_train = y[:-1]
    X_train = X[:-1]
    
    y_test = y[-1]
    X_test = X[-1].reshape(-1, 1)
    
    reg = reg.fit(X_train, y_train)

    reg.coef_
    reg.intercept_

    y_hat = reg.predict(X_test)
    res_train = y_train - reg.predict(X_train)
    res_test = y_test - y_hat
    
    change_flag[i] = (res_test > 2*np.std(res_train))
    
#    plt.plot(X ,y, 'r-',X_test, y_hat, '*',linewidth=2)

change_flag = change_flag.astype(bool)
np.sum(change_flag)
ax2 = df1.plot(figsize=(10, 8), marker='o', linewidth=1)

df2 = df1.loc[change_flag,:]

for d in df2.index:
    ax2.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)

    
ax.plot(df1.index[change_flag], df1.values[change_flag], 'o')









    
    
    
    
    
    
    
    
    
    