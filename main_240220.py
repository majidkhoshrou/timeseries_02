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

from sklearn.preprocessing import minmax_scale

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
ts1[0]=5
ts1 = np.cumsum(ts1)
idx1 = pd.date_range('2018-01-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(10).mean().fillna(method='bfill')
df1.iloc[25]=17
df1.iloc[75:85]=3
#df1.iloc[85:]=12

df1.ts = minmax_scale(df1.ts + np.random.randn(n1)/20)
df1.plot(figsize=(10, 8), marker='o', linewidth=1)
###############################################################################
# develope a model

windowLength = 3

df1['shift_01'] = df1.ts.shift(periods=1)
#df1['shift_02'] = df1.ts.shift(periods=2)

df1['diff_01'] = df1.shift_01.diff(1)
#df1['diff_02'] = df1.shift_01.diff(2)

#df1['quantile_10']=df1.shift_01.rolling(windowLength).quantile(0.1,interpolation='midpoint')
#df1['quantile_50']=df1.shift_01.rolling(windowLength).quantile(0.5,interpolation='midpoint')
#df1['quantile_90']=df1.shift_01.rolling(windowLength).quantile(0.9,interpolation='midpoint')


df1 = df1.replace([np.inf, -np.inf], np.nan)
df1.fillna(method='bfill', inplace=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

n_tree=windowLength
rf = RandomForestRegressor(n_estimators=n_tree, max_depth=2, random_state=0, criterion='mae', max_features=None)
#rf = GradientBoostingRegressor(loss='huber',n_estimators=n_tree)

startDate = pd.to_datetime('January 3, 2018')
endDate = pd.to_datetime('April 10, 2018')

d1 = startDate - datetime.timedelta(days=1)
df2 = df1.copy()
df2['ts_hat'] = np.nan
#df2['std_err'] = np.nan
df2['residuals'] = 0
feature_importance = []

#df2['outside_boundary']=np.nan

for d in pd.date_range(startDate, endDate):
    
    d1 = d - datetime.timedelta(windowLength+1)
    
    d2 = d - datetime.timedelta(1)
    
    X_train, y_train = df2.loc[d1:d2,df2.columns.difference(['ts', 'ts_hat','residuals'])], df2.loc[d1:d2,'ts']
    
    X_test, y_test = df2.loc[d,df2.columns.difference(['ts', 'ts_hat','residuals'])], df2.loc[d,'ts']
    
    try:
        rf.fit(X_train, y_train)
        
    except Exception as e:
        print(e)
    res_train = y_train - rf.predict(X_train)
    res_train.std()
#    df2.loc[d, 'std_err'] = np.std(rf.predict(X_train)-y_train)
    y_hat = rf.predict(X_test.values.reshape(1,-1)) 
    df2.loc[d,'ts_hat'] = y_hat
    res = y_test - y_hat
    np.abs(res-res_train.mean()) > 2*y_train.std()
    df2.loc[d,'residuals'] = res
    feature_importance.append(rf.feature_importances_)


#    rf.estimators_[0].predict
#    per_tree_pred  = [ tree.predict(X_test) for tree in rf.estimators_ ]
#    per_tree_pred = np.array(per_tree_pred)
#    np.percentile(per_tree_pred, [5,25,50,75,95], axis=0)[:,0]

from scipy.stats import zscore


df3 = df2.loc[:,['ts','ts_hat','residuals']]  
# subplots=True, layout = (2,3),
ax3 = df3.plot(figsize=(30, 20), marker='o', linewidth=3)

def getZscore(x):
    return zscore(x)[-1]

idx = df3.residuals.diff().rolling(14).apply(getZscore, raw=True).abs()>2.5
df4 = df3.loc[idx.values,:]

for d in df4.index:
    ax3.axvspan(d-datetime.timedelta(1),d, color = 'y', alpha=.2)



#plt.savefig('anomaly_rf.pdf')
# subplots=True,layout = (3,1),     

a = np.array(feature_importance).T
att = list(X_train.columns)
b = pd.DataFrame(dict(zip(att, a)))
    
#b.plot(figsize=(10, 8), marker='o', linewidth=1)
#plt.savefig('att_imp.pdf')



    
    
    
    
    
    
    
    
    
    