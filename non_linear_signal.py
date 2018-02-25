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
from glmnet import ElasticNet

import os as os


def get_linear_sig(sig_sr, ret_sr, look_back = 12, fit_intercept = False):
    sig_sr = sig_sr.copy()
    ret_sr = ret_sr.copy()
    
    rebalance_dates = (sig_sr.index.unique()).sort_values()
    data = pd.DataFrame()
    data['X'] = sig_sr
    data['y'] = ret_sr
    sig_sr = pd.Series()
    
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
        this_sig = model.predict(test_x)
        this_sig = pd.Series(this_sig, index = test_x.index)
        this_sig = this_sig / np.sum( np.abs( this_sig ) )
        #this_sig = this_sig.rank()
        #this_sig = ( this_sig - this_sig.mean() ) / this_sig.std()
        #this_sig = this_sig.fillna(0.0)
        sig_sr = sig_sr.append(this_sig)
    
    return(sig_sr)


def get_knn_non_linear_sig(sig_sr, ret_sr, look_back = 12,num_buckets=10):
    sig_sr = sig_sr.copy()
    ret_sr = ret_sr.copy() 
    
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
        this_nl_sig = this_nl_sig / np.sum(np.abs(this_nl_sig))
        #this_nl_sig = this_nl_sig.rank()
        
        #this_nl_sig = ( this_nl_sig - this_nl_sig.mean() ) / this_nl_sig.std()      
        #this_nl_sig = this_nl_sig.fillna(0.0)
        nl_sig_sr = nl_sig_sr.append(this_nl_sig)
    
    return(nl_sig_sr)

def get_kernel_non_linear_sig(sig_sr, ret_sr, look_back = 12,
                              alpha=0.5, gamma=0.5):
    sig_sr = sig_sr.copy()
    ret_sr = ret_sr.copy()
    
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
        
        model = KernelRidge(kernel='rbf',alpha=alpha,gamma= gamma)
        model.fit(train_x,train_y)
        this_nl_sig = model.predict(test_x)
        this_nl_sig = pd.Series(this_nl_sig, index = test_x.index)
        this_nl_sig = this_nl_sig / np.sum(np.abs(this_nl_sig))
        #this_nl_sig = this_nl_sig.rank()
        
        #this_nl_sig = ( this_nl_sig - this_nl_sig.mean() ) / this_nl_sig.std()      
        #this_nl_sig = this_nl_sig.fillna(0.0)  

        nl_sig_sr = nl_sig_sr.append(this_nl_sig)
    
    return(nl_sig_sr)


def get_lambda(model, num_signals= 5):
    coef_path = model.coef_path_
    tol = 1e-8
    
    num_coef = [ (np.where( np.abs(coef_path[:,i]) > tol )[0]).shape[0] for i in range(coef_path.shape[1]) ]
    num_coef = np.array(num_coef)
    #print(num_coef)
    coef_ind = ( np.abs(num_coef - num_signals )).argmin()
    coef_w = coef_path[:,coef_ind]
    w_ind = np.where( np.abs(coef_w) > tol )[0]
    s = model.lambda_path_[coef_ind]
    return(s, w_ind)

def get_glmnet_sig(sig_df, ret_sr, look_back = 12,num_sig_vec =[5], alpha = 0.5 ):
    sig_df = sig_df.copy()
    ret_sr = ret_sr.copy()
    
    rebalance_dates = (sig_df.index.unique()).sort_values()
    
    data = sig_df
    data['y'] = ret_sr
    comb_sig_df = pd.DataFrame()
    sel_sig_names_vec = []
    print('inside')    
    
    for ind in range(look_back, rebalance_dates.shape[0] ) :

        r_d = rebalance_dates[ind]
        print(r_d)
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

        train_x = train_data.drop(['y'], axis=1)
        train_y = train_data['y']
        test_x = test_data.drop(['y'], axis=1)
        test_y = test_data['y']
        
        model = ElasticNet(alpha=alpha, fit_intercept=True, n_lambda=1000,tol=1e-8 )
        model.fit(train_x,train_y)
        
        this_comb_sig_df = pd.DataFrame()
        sel_sig_names = []
        
        for num_sig in num_sig_vec :
            s, w_ind = get_lambda(model, num_sig)
            #print(s)
            #print(w_ind)
            this_sel_sig = test_x.columns[w_ind].values
            if this_sel_sig.shape[0] < num_sig :
                this_sel_sig = np.append( this_sel_sig, 
                                         ['NA']*(num_sig - this_sel_sig.shape[0]) )
            #print(this_sel_sig)
            this_sig = model.predict(test_x, s)       
            this_sig = pd.Series(this_sig, index = test_x.index)
            this_sig = this_sig / np.sum( np.abs( this_sig ) )
                      
            this_comb_sig_df[str(num_sig) ] = this_sig
            sel_sig_names.append(this_sel_sig)
        
        sel_sig_names_vec.append(sel_sig_names)
        this_comb_sig_df.index = test_x.index
        #this_comb_sig_df = this_comb_sig_df.rank(axis=0)
        #this_comb_sig_df = ( this_comb_sig_df - this_comb_sig_df.mean(axis=0) ) / this_comb_sig_df.std(axis=0)      
         
        
        comb_sig_df = comb_sig_df.append(this_comb_sig_df)
    
    return(comb_sig_df, sel_sig_names_vec)
