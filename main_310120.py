# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:00:36 2020

@author: MajidKhoshrou
"""

# import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

plt.style.available

plt.style.use('fivethirtyeight')

# generate timeseries
n1 = 400
n2 = 200
sigma = 10
mu = .1

ts1 = np.random.randn(n1)
ts1[0]=2
ts1 = np.cumsum(ts1)

ts2 = np.cumsum(sigma*np.random.randn(n2)+mu)

#ts1[100:107] = 30+ts1[100:107]
#thresh = 5
#ts1 = ts1 - thresh
##ts1[50]=20
#
idx1 = pd.date_range('2018-01-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)

df_new = pd.concat([df1, df1.pct_change(1)], axis=1)
df_new.columns = ['ts', 'pct change']
df_new = df_new.replace([np.inf, -np.inf], np.nan)
df_new.dropna(inplace=True)
df_new.plot(linewidth=2, figsize=(10, 8), subplots=True, layout = (2,1))

plt.savefig("zero_crossing.png")
gggg

#idx1 = pd.date_range('2018-01-01', periods=n1+n2, freq='D')
#df1 = pd.DataFrame({"ts": np.concatenate((ts1,ts2)) }, index=idx1)


ax1 = df1.plot(linewidth=2, figsize=(10, 8)) # , figsize=(10, 8), linewidth=2, fontsize=6

#ax1.axvspan('2019 August','2019 September', color = 'y', alpha=.2)
plt.savefig("zerocrossing01.png")

df1_pct_change_01 = df1.pct_change(1).replace([np.inf, -np.inf], np.nan)
df1_pct_change_01.dropna(inplace=True)

ax11 = df1_pct_change_01.plot(linewidth=2, figsize=(10, 8))
plt.savefig("zerocrossing02.png")

poi =  df1_pct_change_01.index[ abs(df1_pct_change_01.ts.values) > 5 ]





pd.concat([df1, df1_pct_change_01])


df1_pct_change_02 = df1.pct_change(2).dropna()


#for i in list(poi):
#    print(i)
#    ax1.axvline(i, color='k', linestyle='--');
#plt.show()

df1_pct_change_01.sort_values(by='ts')

ax12 = df1_pct_change_02.plot()


# to detect the upward and downward trend we can get the mode of the sign of 
# the 2nd derivatives. Here sum is negative so downward trend!

WindowOfObs = df1['2018 January':]
WindowOfObs.reset_index(inplace=True)
mod_val = np.polyfit(WindowOfObs.index, WindowOfObs.ts,1)
if mod_val[0] > 0:
    print("upward trend")
elif mod_val[0] <0:
    print("downward trend")
else:
    print("stationary!")
    
####
ts2 =np.random.randn(n2)
ts2[0]=0
ts2 = np.cumsum(ts2)
ts2[100:106] = 20+ts2[100:106]
ts2[50:53] = -20+ts2[50:53]
idx2 = pd.date_range(df1.index[-1]+datetime.timedelta(days=1), periods=n2, freq='D')
df2 = pd.DataFrame({"ts": ts2 }, index=idx2)
#
# plt.plot(df1)
# ax2 = df2.plot()
new_df = pd.concat([df2,df2.diff()], axis=1)
new_df.columns = ['ts','diff']
new_df.plot(subplots=True, layout = (2,1), linewidth=2, figsize=(10, 8))
plt.savefig("anomaly03.png")

# line ax.axhline
# region ax.axhspan
    
# rolling window, median, outlier
    
# df.rolling(window=7).median() or .mean() +/- 2 * (df.rolling(window=7).std())
    
# trends, anomalies and noise
# seasonal affective disorder

#################
# plot timeseries on individual plots...
#meat.plot(subplots=True, 
#          layout=(2,4), 
#          sharex=False, 
#          sharey=False, 
#          colormap='viridis', 
#          fontsize=2, 
#          legend=False, 
#          linewidth=0.2)
#
#plt.show()
######################




    










