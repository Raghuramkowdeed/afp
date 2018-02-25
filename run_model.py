# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:32:16 2018

@author: raghuramkowdeed
"""

import os as os
import pandas as pd
import numpy as np
import datetime as dt
import multiprocessing as mp
import tempfile

exec(open("./train_model.py").read())


def run_model(sig_df, ret_sr, model_name, train_model_arg=None, 
              pred_model_arg ={},look_back=12) :
                  
    sig_df = sig_df.copy()
    ret_sr = ret_sr.copy()
    
    rebalance_dates = (sig_df.index.unique()).sort_values()
    data = sig_df
    data['ret'] = ret_sr
    
    model_sig_sr = pd.Series()    
    
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

        train_X = train_data.drop(['ret'], axis=1)
        train_y = train_data['ret']
        test_X = test_data.drop(['ret'], axis=1)
        test_y = test_data['ret']
        
        model = train_model(train_X, train_y, model_name, train_model_arg)

        this_sig = predict_model(model_name, model, test_X, pred_model_arg )
        this_sig = pd.Series(this_sig, index = test_X.index)
        this_sig = this_sig / np.sum( np.abs( this_sig ) )
        #this_sig = this_sig.rank()
        #this_sig = ( this_sig - this_sig.mean() ) / this_sig.std()
        #this_sig = this_sig.fillna(0.0)
        model_sig_sr = model_sig_sr.append(this_sig)
    
    return(model_sig_sr)


def this_train_model_func(this_arg):
    exec(open("./train_model.py").read())
    i =0
    model_name = this_arg[i]
    i=i+1
    look_back = this_arg[i]
    i=i+1
    train_model_arg = this_arg[i]
    i=i+1
    pred_model_arg = this_arg[i]
    i=i+1
    ind = this_arg[i]
    i=i+1
    data_file = this_arg[i]
    i=i+1
    out_file = this_arg[i]
    i=i+1
    
    data = pd.read_pickle(data_file)
    rebalance_dates = (data.index.unique()).sort_values()
    
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

    train_X = train_data.drop(['ret'], axis=1)
    train_y = train_data['ret']
    test_X = test_data.drop(['ret'], axis=1)
    test_y = test_data['ret']
        
    model = train_model(train_X, train_y, model_name, train_model_arg)

    this_sig = predict_model(model_name, model, test_X, pred_model_arg )
    this_sig = pd.Series(this_sig, index = test_X.index)
    this_sig = this_sig / np.sum( np.abs( this_sig ) )
    this_sig.to_pickle(out_file)
    
    return(0)
    

def run_model_parallel(sig_df, ret_sr, model_name, train_model_arg=None, 
              pred_model_arg ={},look_back=12, num_cores=2) :
                  
    sig_df = sig_df.copy()
    ret_sr = ret_sr.copy()
    
    rebalance_dates = (sig_df.index.unique()).sort_values()
    data = sig_df
    data['ret'] = ret_sr
    fd, all_data_file = tempfile.mkstemp()    
    data.to_pickle(all_data_file)
    
    model_sig_sr = pd.Series()    
    pred_y_files = []
    
    train_arg_vec = []
    for ind in range(look_back, rebalance_dates.shape[0] ) :
        this_arg = []
        this_arg.append(model_name)
        this_arg.append(look_back)
        this_arg.append(train_model_arg)
        this_arg.append(pred_model_arg)
        this_arg.append(ind)


        this_arg.append(all_data_file)
        
        fd, this_out_file = tempfile.mkstemp()
        this_arg.append(this_out_file)    
        pred_y_files.append(this_out_file)
        
        train_arg_vec.append(this_arg)

        p = mp.Pool(num_cores)
        res = p.map(this_train_model_func,train_arg_vec)

    for ind in range(look_back, rebalance_dates.shape[0] ) :
        this_sig = pd.read_pickle(pred_y_files[ind])
        model_sig_sr = model_sig_sr.append(this_sig)   
        
    return(model_sig_sr)
    
