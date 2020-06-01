# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:20:10 2020

@author: MajidKhoshrou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.linear_model  import LinearRegression


plt.style.available

plt.style.use('fivethirtyeight')

# generate timeseries
n1 = 100
n2 = 200
sigma = 10
mu = .1

ts1 = np.random.randn(n1)
ts1[0]=0
ts1 = np.cumsum(ts1)

ts2 = np.cumsum(sigma*np.random.randn(n2)+mu)

#ts1[100:107] = 30+ts1[100:107]
#thresh = 5
#ts1 = ts1 - thresh
##ts1[50]=20
#
idx1 = pd.date_range('2018-01-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(7).mean()
df1.plot()


df_new = pd.concat([df1, df1.pct_change(1)], axis=1)
df_new.columns = ['ts', 'pct change']
df_new = df_new.replace([np.inf, -np.inf], np.nan)
df_new.dropna(inplace=True)
df_new.plot(linewidth=2, figsize=(10, 8), subplots=True, layout = (2,1))

# start with window

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta #rho

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

nr_Obs = 7
nr_delta = datetime.timedelta(nr_Obs)
slopes = []
ang = []
for d2 in idx1:
    d1 = d2 - nr_delta
    
    ts = np.empty((nr_Obs,))
    ts[:] = np.nan
    
    WindowOfObs = df1[d1:d2]
    WindowOfObs.reset_index(inplace=True)
    try:
        x = df1.loc[d2-datetime.timedelta(7)].ts
        y = df1.loc[d2].ts
    except:
        x=np.nan
        y=np.nan
    ang.append(cart2pol(x, y))
    


    try:
        slopes.append(np.polyfit(WindowOfObs.index, WindowOfObs.ts,1)[0])
        
    except Exception as e:
        print()
        slopes.append(np.nan)
        
df_slopes = pd.DataFrame({"slopes": slopes }, index=idx1)
df_signs = pd.DataFrame({"signs": np.sign(slopes) }, index=idx1)
df_theta = pd.DataFrame({"theta": ang }, index=idx1)

df_new = pd.concat([df1,df_slopes,df_signs ], axis=1)
df_new.columns = ['timeseries', 'slope', 'sign' ]
df_new = df_new.replace([np.inf, -np.inf], np.nan)
df_new.dropna(inplace=True)
df_new.plot(linewidth=2, figsize=(20, 15), subplots=True, layout = (3,1))
#plt.savefig("trendchanges.png")

#df_theta.plot()

#y= np.array([3,4,5,8])
#x = np.array(list(range(len(y))))
#
#np.polyfit(x,y,1)
#    


#from sklearn import linear_model
#from scipy import stats
#import numpy as np
#
#class LinearRegression(linear_model.LinearRegression):
#    """
#    LinearRegression class after sklearn's, but calculate t-statistics
#    and p-values for model coefficients (betas).
#    Additional attributes available after .fit()
#    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
#    which is (n_features, n_coefs)
#    This class sets the intercept to 0 by default, since usually we include it
#    in X.
#    """
#
#    def __init__(self, *args, **kwargs):
#        if not "fit_intercept" in kwargs:
#            kwargs['fit_intercept'] = False
#        super(LinearRegression, self).__init__(*args, **kwargs)
#
#    def fit(self, X, y, n_jobs=1):
#        self = super(LinearRegression, self).fit(X, y, n_jobs)
#
#        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
#        se = np.array([
#            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
#                                                    for i in range(sse.shape[0])
#                    ])
#
#        self.t = self.coef_ / se
#        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
#        return self
#
#lm = LinearRegression(x,y)
#
#
#import statsmodels.api as sm
#
#x = sm.add_constant(x)
#model = sm.OLS(y,x)
#model.
#results = model.fit()
#results.f_pvalue
#    



