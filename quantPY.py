# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:45:57 2017

@author: Yohan Reyes
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import kurtosis
import math
import csv
import scipy as sp
from scipy.stats import norm
from tqdm import tqdm

from datetime import datetime
# from pandas_datareader import data, wb
from numpy import cumsum, log, polyfit, sqrt, std, subtract, mean, log10
from numpy.random import randn
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as ts

def movavg(Asset,n_days):
    Asset = pd.DataFrame(Asset);
    moving_avg = [];
#    moving_avg = pd.DataFrame(moving_avg)
#    moving_avg = [0,0,0,0];
#    moving_avg = pd.DataFrame.as_matrix(moving_avg)
#    moving_avg = np.empty([Asset.shape[0], Asset.shape[1]])
#    moving_avg = np.zeros(shape=(Asset.shape[0],Asset.shape[1]))
#   list(my_dataframe.columns.values)
    moving_avg = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=list(Asset));
    Asset = pd.DataFrame.as_matrix(Asset);
    i2 = 0;
    n = float(n_days);
    for i1 in range(n_days,len(Asset)+1):
        moving_avg.iloc[i1-1,:] = (sum(Asset[i2:i1,:]))/n;
        i2 = i2+1;
#        print str(i1)+'-'+str(i2)
        
    return moving_avg


def multi_movavg(Asset,vector_days):
    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_days:   
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' ' + str(i2))
            i1 = i1+1;
      
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp = [];
    temp = pd.DataFrame(temp);
    mov = [];
    mov = pd.DataFrame(mov);
    
    for i in vector_days:
        if i == vector_days[0]:
            mov = movavg(Asset,i);
        else: 
            temp = movavg(Asset,i);
            mov = pd.concat([mov, temp], axis=1)
    
    mov.columns = cols;
    
    return mov

def accumulated_kurt(Asset):
    Asset = pd.DataFrame(Asset);
    
    i2 = 0
    cols = [];
    for i1 in list(Asset):
        cols.append(list(Asset)[i2]+' kurt');
        i2 = i2+1;

    acc_kurt = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=list(Asset));
    Asset = pd.DataFrame.as_matrix(Asset);
    
    acc_kurt.columns = cols;
    i2 = 0;
    for i in range(4,len(Asset)+1):
        acc_kurt.iloc[i-1,:] = kurtosis(Asset[0:i,:])
        i2 = i2+1

    return acc_kurt

##############################################################################

'''
def accumulated_adf(Asset):
    Asset = pd.DataFrame(Asset);
    
    i2 = 0
    cols = [];
    for i1 in list(Asset):
        cols.append(list(Asset)[i2]+' kurt');
        i2 = i2+1;

    acc_kurt = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=list(Asset));
    Asset = pd.DataFrame.as_matrix(Asset);
    
    acc_kurt.columns = cols;
    i2 = 0;
    for i in range(4,len(Asset)+1):
        acc_kurt.iloc[i-1,:] = kurtosis(Asset[0:i,:])
        i2 = i2+1

    return acc_kurt
'''

'''
def multi_accumulated_kurt_window(Asset,vector_windows):

    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_windows:
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' ' + str(i2))
            i1 = i1+1;
    
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp = [];
    temp = pd.DataFrame(temp);
    acc_kurt_win = [];
    acc_kurt_win = pd.DataFrame(acc_kurt_win);
    
    for i in vector_windows:
        if i == vector_windows[0]:
            acc_kurt_win = accumulated_kurt_window(Asset,i);
        else: 
            temp = accumulated_kurt_window(Asset,i);
            acc_kurt_win = pd.concat([acc_kurt_win, temp], axis=1)
    
    acc_kurt_win.columns = cols;
    
    return acc_kurt_win
'''

##############################################################################

def accumulated_kurt_window(Asset,window):
    Asset = pd.DataFrame(Asset);
    
    i2 = 0
#    cols = [];
    
    '''
    for i1 in list(Asset):
        cols.append(list(Asset)[i2]+' accumulated kurt ' + str(window));
        i2 = i2+1;
    '''    
    
    acc_kurt_window = pd.DataFrame(0, index=np.arange(Asset.shape[0]), columns=list(Asset));
    Asset = pd.DataFrame.as_matrix(Asset);
    
    window = int(window);
#     acc_kurt_window.columns = cols;
    i2 = 0;
    
    for i in range(window,len(Asset)+1):
        acc_kurt_window.iloc[i-1,:] = kurtosis(Asset[i2:i,:])
        i2 = i2+1
    
    return acc_kurt_window
    

def multi_accumulated_kurt_window(Asset,vector_windows):

    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_windows:
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' ' + str(i2))
            i1 = i1+1;
    
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp = [];
    temp = pd.DataFrame(temp);
    acc_kurt_win = [];
    acc_kurt_win = pd.DataFrame(acc_kurt_win);
    
    for i in vector_windows:
        if i == vector_windows[0]:
            acc_kurt_win = accumulated_kurt_window(Asset,i);
        else: 
            temp = accumulated_kurt_window(Asset,i);
            acc_kurt_win = pd.concat([acc_kurt_win, temp], axis=1)
    
    acc_kurt_win.columns = cols;
    
    return acc_kurt_win


def logret(Asset,n_days):
    Asset = pd.DataFrame(Asset);
    n_days = int(n_days)
    log_ret = np.log(Asset)-np.log(Asset.shift(periods=n_days, freq=None, axis=0))
    return log_ret



def logret_multi(Asset,vector_days):
    Asset = pd.DataFrame(Asset);
    cols = [];
    
    i1 = 0;
    for i2 in vector_days:
        i1 = 0
        for i in list(Asset):
            cols.append(list(Asset)[i1]+' ' + str(i2))
            i1 = i1+1;
    
    i1 = 0;
    i2 = Asset.shape[1]-1
    
    temp = [];
    temp = pd.DataFrame(temp);
    log_ret_multi = [];
    log_ret_multi = pd.DataFrame(log_ret_multi);
    
    for i in vector_days:
        if i == vector_days[0]:
            log_ret_multi = logret(Asset,i);
        else: 
            temp = logret(Asset,i);
            log_ret_multi = pd.concat([log_ret_multi, temp], axis=1)
    
    log_ret_multi.columns = cols;
    
    return log_ret_multi

def perc_ret(Asset):
    Asset = pd.DataFrame(Asset);
    percentage_ret = (Asset)/(Asset.shift(periods=1, freq=None, axis=0))
    percentage_ret = percentage_ret-1
    return percentage_ret

#%% Fractals

def hurst_RS(ts, plots):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, len(ts))

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses 
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    if plots == True:
        # plot on log-log scale
        plt.figure()
        plt.plot(log(lags), log(tau))

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


# more useful, maybe 
def rs(Z):

#     start_time = time.clock()
    # took from matlab
    Z = pd.DataFrame(Z)
    Z = pd.DataFrame.as_matrix(Z)
    
    m=len(Z)
    x=[None]*m
    y=[None]*m
    y2=[None]*m
    
#    start_time = time.clock()

    for tau in range(2,m):
#    for tau in tqdm(range(2,m), ascii=True, desc='Hurst exp'):
        X=[None]*(tau+1)
        Zsr=mean(Z[0:tau+1])
    
        for t in range(0,tau+1):
            X[t]=float(sum(Z[0:t+1]-Zsr))
    
        R=max(X)-min(X)
        S=std(Z[0:tau+1])
        H=log10(R/float(S))/log10((tau+1)/2.0)
        
        x[tau]=log10(tau+1)
        y[tau]=H
        y2[tau]=log10(R/float(S))

#    print(-start_time + time.clock())
    return H, y2, y

# http://epchan.blogspot.fr/2016/04/mean-reversion-momentum-and-volatility.html

def hurst_ernie_chan(p):

    variancetau = []; tau = []

    for lag in lags:  
        #  Write the different lags into a vector to compute a set of tau or lags  
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns  
        # call this pp or the price difference  
        pp = subtract(p[lag:], p[:-lag])  
        variancetau.append(var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.  
    #print tau  
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope  
    m = polyfit(log10(tau),log10(variancetau),1)

    hurst = m[0] / 2

    return hurst  


def sharpe(series):
    ret = numpy.divide(numpy.diff(series),series[:-1])
    return(numpy.mean(ret)/numpy.std(ret))


#%%

def adft_window(Asset, lag):
    Asset = pd.DataFrame(Asset)
    window_pval = []
    adft = []
    rejected = []
    difference = []
    # samples = pd.DataFrame.as_matrix(samples)
    i1 = 0
    for i in tqdm(range(lag-1,len(Asset))):
        adft_temp = ts.adfuller(Asset.iloc[i1:i,0], maxlag=None, regression='ctt', autolag='AIC', store=False, regresults=False)
        adft.append(adft_temp)
        window_pval.append(adft_temp[1])
        if 0.05<=adft_temp[1]:
            rejected.append(0)
            difference.append(adft_temp[1]-0.05)
        elif 0.5>adft_temp[1]>=(-0.05):
            rejected.append(0.5)
            difference.append(-adft_temp[1]+adft_temp[4]['10%'])
        if (adft_temp[4]['10%']>adft_temp[1]) & (adft_temp[4]['5%']>=adft_temp[1]):
            rejected.append(5)
            difference.append(-adft_temp[1]+adft_temp[4]['5%'])
        elif (adft_temp[4]['5%']>adft_temp[1]) & (adft_temp[4]['1%']>=adft_temp[1]):
            rejected.append(3)
            difference.append(-adft_temp[1]+adft_temp[4]['1%'])
        else:
            rejected.append(1)
            difference.append(adft_temp[1])
        i1 = i1+1

    return window_pval, adft, rejected, difference



#%% Autocorr of the TS using the number of days that give more corr in the log return

def autocorr(Asset,n_days):
    
    cols = []

    for i in Asset.columns:
        cols.append(str(i)+' corr '+str(n_days))

    Asset1 = Asset.shift(periods=n_days, freq=None, axis=0).copy()
    Asset1.columns = list(cols)
    auto = pd.concat([Asset,Asset1],axis = 1)

    auto_corr = auto.corr()
    # sns.heatmap(auto_corr)
    return auto_corr
    








