# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:18:34 2018

@author: raghuramkowdeed
"""

import pandas as pd
import numpy as np
import datetime as dt
import dateutil as du
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import multiprocessing as mp
import os as os


def get_linear_sig(sig_sr, ret_sr, look_back = 12, fit_intercept = True):
    
    rebalance_dates = (sig_sr.index.unique()).sort_values()
    data = pd.DataFrame()
    data['X'] = sig_sr
    data['y'] = ret_sr
    nl_sig_sr = pd.Series()
    
    for ind in range(look_back, rebalance_dates.shape[0] ) :

        r_d = rebalance_dates[ind]
        train_end_date = rebalance_dates[ind-1] 
        #train_end_year = train_end_date.year

        train_start_date = rebalance_dates[ind-look_back]
        #train_start_year = train_start_date.year

        #curr_year = r_d.year

        train_data = data
        train_data = train_data[train_data.index>=train_start_date]
        train_data = train_data[train_data.index<=train_end_date]

        test_data = data
        test_data = test_data[test_data.index==r_d]

        train_x = train_data[ ['X']]
        train_y = train_data['y']
        test_x = test_data[['X']]
        test_y = test_data['y']
        
        model = LinearRegression(fit_intercept= fit_intercept)
        model.fit(train_x,train_y)
        this_nl_sig = model.predict(test_x)
        this_nl_sig = pd.Series(this_nl_sig, index = test_x.index)
        nl_sig_sr = nl_sig_sr.append(this_nl_sig)
    
    return(nl_sig_sr)


def get_knn_non_linear_sig(sig_sr, ret_sr, look_back = 12,num_buckets=10):
    
    rebalance_dates = (sig_sr.index.unique()).sort_values()
    data = pd.DataFrame()
    data['X'] = sig_sr
    data['y'] = ret_sr
    nl_sig_sr = pd.Series()
    
    for ind in range(look_back, rebalance_dates.shape[0] ) :

        r_d = rebalance_dates[ind]
        train_end_date = rebalance_dates[ind-1] 
        #train_end_year = train_end_date.year

        train_start_date = rebalance_dates[ind-look_back]
        #train_start_year = train_start_date.year

        #curr_year = r_d.year

        train_data = data
        train_data = train_data[train_data.index>=train_start_date]
        train_data = train_data[train_data.index<=train_end_date]

        test_data = data
        test_data = test_data[test_data.index==r_d]

        train_x = train_data[ ['X']]
        train_y = train_data['y']
        test_x = test_data[['X']]
        test_y = test_data['y']
        
        model = KNeighborsRegressor(n_neighbors=int(train_data.shape[0]*1.0/num_buckets))
        model.fit(train_x,train_y)
        this_nl_sig = model.predict(test_x)
        this_nl_sig = pd.Series(this_nl_sig, index = test_x.index)
        nl_sig_sr = nl_sig_sr.append(this_nl_sig)
    
    return(nl_sig_sr)

def get_kernel_non_linear_sig(sig_sr, ret_sr, look_back = 12,gamma=0.1):
    
    rebalance_dates = (sig_sr.index.unique()).sort_values()
    data = pd.DataFrame()
    data['X'] = sig_sr
    data['y'] = ret_sr
    nl_sig_sr = pd.Series()
    
    for ind in range(look_back, rebalance_dates.shape[0] ) :

        r_d = rebalance_dates[ind]
        #print(r_d)
        train_end_date = rebalance_dates[ind-1] 
        #train_end_year = train_end_date.year

        train_start_date = rebalance_dates[ind-look_back]
        #train_start_year = train_start_date.year

        #curr_year = r_d.year

        train_data = data
        train_data = train_data[train_data.index>=train_start_date]
        train_data = train_data[train_data.index<=train_end_date]

        test_data = data
        test_data = test_data[test_data.index==r_d]

        train_x = train_data[ ['X']]
        train_y = train_data['y']
        test_x = test_data[['X']]
        test_y = test_data['y']
        
        model = KernelRidge(kernel='rbf', gamma = gamma)
        model.fit(train_x,train_y)
        this_nl_sig = model.predict(test_x)
        this_nl_sig = pd.Series(this_nl_sig, index = test_x.index)
        nl_sig_sr = nl_sig_sr.append(this_nl_sig)
    
    return(nl_sig_sr)
