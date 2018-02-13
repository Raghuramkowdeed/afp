#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:28:15 2018

@author: raghuramkowdeed
"""

import pandas as pd
import numpy as np
import datetime as dt
import dateutil as du

execfile('./models/quintile_model.py')

def non_linearity_test(data, sig_name='acc', 
                       start_year=1990, end_year=2015, 
                       look_back=5, num_buckets=10):
    
    l_ic_vec = []
    nl_ic_vec = []
    comb_ic_vec = []
    nl_l_corr_vec = []
    model_vec = []
    
    start_date = dt.datetime.strptime(str(start_year)+'0101','%Y%m%d')
    end_date = dt.datetime.strptime(str(end_year)+'0101','%Y%m%d')
    
    rebalance_dates = []
    curr_date = start_date
    
    while curr_date <= end_date :
        rebalance_dates.append(curr_date)
        curr_date = curr_date +  du.relativedelta.relativedelta(years=1)
    
   
    for r_d in rebalance_dates :
        train_end_date = r_d - du.relativedelta.relativedelta(years=1) 
        train_end_year = train_end_date.year
        
        train_start_date = train_end_date - du.relativedelta.relativedelta(years=look_back)
        train_start_year = train_start_date.year
        
        curr_year = r_d.year
        
        train_data = data
        train_data = train_data[train_data['fyear']>=train_start_year]
        train_data = train_data[train_data['fyear']<=train_end_year]
        
        test_data = data
        test_data = test_data[test_data['fyear']==curr_year]
    
        
        model = QuintileSort(num_buckets,'median')
    
        train_x = train_data[sig_name]
        train_y = train_data['RET']
        #print(r_d)
        #print(train_x.shape)
        #print(train_y.shape)
        #print('----------------')
    
        model.fit(train_x,train_y)
    
        #model.ret_profile.plot(kind='bar')
    
        test_x = test_data[sig_name]
        test_y = test_data['RET']
    
        nl_pred_ret = model.predict(test_x)
        nl_pred_ret = nl_pred_ret/nl_pred_ret.std()
        
        nl_corr = nl_pred_ret.corr(test_y)
        
        beta = train_x.corr(train_y) * train_y.std() / train_x.std()
        l_pred_ret = test_x*beta
        l_pred_ret = l_pred_ret/l_pred_ret.std()
        
        l_corr = l_pred_ret.corr(test_y)
    
        comb_pred_ret = ( nl_pred_ret + l_pred_ret )/2.0
        comb_corr = comb_pred_ret.corr(test_y)
        
        
        nl_l_corr = nl_pred_ret.corr(l_pred_ret)
        
        
        l_ic_vec.append(l_corr)
        nl_ic_vec.append(nl_corr)
        
        comb_ic_vec.append(comb_corr)
        
        nl_l_corr_vec.append(nl_l_corr)
        
        model_vec.append(model)
    
    
    #print( r_d,l_corr, nl_corr )
    
    df = pd.DataFrame()
    df['l_corr'] = l_ic_vec
    df['nl_corr'] = nl_ic_vec
    df['comb_ic'] = comb_ic_vec
    df['nl_l_corr'] = nl_l_corr_vec

    df.index = rebalance_dates
    
    return df, model_vec