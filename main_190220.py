# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:15:04 2020

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

df = pd.read_csv('data/keyhole_target_emotion_170220.csv', index_col='Created_timestamp', parse_dates=['Created_timestamp'])
df.sort_index( inplace=True)
assert all(df.sort_index().index == df.index)

#df = df.groupby(['search_terms','platforms','Date'])
#.fillna(method='ffill').round(2)
#df.reset_index(inplace=True)
#sns.lineplot(x='Date', y='mood', hue = 'search_terms', data=df)
#plt.savefig('mood_hue.png')


df1 = df.loc[(df.search_terms=='philips') & (df.platforms=='all')]

df2 = df1[['mood']]
df2 = df2.resample('D').mean().fillna(method='bfill')
#interpolate(method ='linear', limit_direction ='forward')

#df2 = (df2-df2.mean())/df2.std()

startDate = pd.to_datetime('January 26, 2020')
endDate = pd.to_datetime('February 15, 2020')

windowObs = df2.loc[startDate:endDate]


#windowObs['normalizd_'] = (windowObs-windowObs.min())/(windowObs.max()-windowObs.min())

print("note: need to do hypothesis checking.")
print("note: consider hypothesis testing for different windows of log return values. ")
print("note: turn 1D into 2D data; do clustering or sth on that!")
print("note: find anomalies (dips like changes) using clustering on distance measures")
print("note: we want to be sensitive to values, normalization may not be necessary.")
print("note: hypothesis testing with and without extreme values, percentiles.")
print("note: detect level changes, positive and negative trend changes.")
print("note: RangePercentile, SlowPosTrend and SlowNegTrend.")
print("note: history window to compute martingale values over the look-back history")
print("note: trade off between sensitinity and confidence.")

quantile_thresh = .1
x =  windowObs.loc[:,'mood']
quantile_vals = x.quantile(q=[quantile_thresh, 1-quantile_thresh])
x < quantile_vals[quantile_thresh] 
x > quantile_vals[1-quantile_thresh]


#interpolate(method ='linear', limit_direction ='forward')
#fillna(method='ffill')

x.plot(linewidth=2, figsize=(20, 15),marker='o')
#plt.savefig('mood_movement.png')

windowObs['shift1'] = windowObs.shift(1)

windowObs['relative'] =\
 2*(windowObs['mood']-windowObs['shift1']).abs()/(windowObs['mood'].abs()+windowObs['shift1'].abs())

windowObs['diff'] = x.diff()
windowObs['log_ret'] = np.log(1+x.pct_change()).replace([np.inf, -np.inf], np.nan)
windowObs['log_ret'].abs() > quantile_thresh

windowObs['log_ret_pct_change'] = windowObs.log_ret.diff()

windowObs_mean = windowObs.mean()
windowObs_std = windowObs.std()
windowObs.loc[windowObs.log_ret.abs()>2*windowObs_std.log_ret]
window_1b4 = df2.loc[startDate:endDate-datetime.timedelta(1)]
window_7b4 = df2.loc[startDate:endDate-datetime.timedelta(7)]

#df2['diff_pct_change'] = df2.movement.diff().pct_change()
#df2['change_flag'] = df2[['diff_pct_change']].abs()>=0.05
##df2.plot(subplots=True, layout = (5,1), linewidth=2, figsize=(10, 8))
##df2.reset_index(inplace=True)
#df2.round(2)
#idx = df2.index[df2['change_flag'].values]
#ax2 = df2['mood'].plot(linewidth=2, figsize=(12, 10))
#for d in idx:
#    ax2.axvspan(d,d+datetime.timedelta(1), color = 'y', alpha=.2)
#plt.savefig('movement_change.png')

x.sort_index(ascending=False).resample('D').interpolate()[::7]

y = windowObs[['mood']]
y['shift1']= y.shift(1)

plt.plot(windowObs['log_ret'],y.mood, 'o')










    










