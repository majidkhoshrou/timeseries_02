# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:02:39 2020

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

n1 = 200
np.random.seed(4)
#ts1 = np.random.randn(n1)

mu=0
sigma=1
ts1 = np.cumsum(sigma*np.random.randn(n1)+mu)

idx1 = pd.date_range('2019-10-01', periods=n1, freq='D')
df1 = pd.DataFrame({"ts": ts1 }, index=idx1)
df1=df1.rolling(window=3, min_periods=1).mean().fillna(method='bfill')


df1.asfreq('M').plot(figsize=(20,15), marker='o', linewidth=3)

pd.Period('2017-01-02')

df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5], 'val':[33,55]})

pd.to_datetime(df.loc[:,'year':'day'])


class Parent:
    def __init__(self):
        print('I am a parent!')

class Child(Parent):
    def __init__(self):
        super().__init__()
        print('I am a child!')
        
