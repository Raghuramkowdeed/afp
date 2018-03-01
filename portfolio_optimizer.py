# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:31:26 2018

@author: raghuramkowdeed
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime as dt
import os

class PortfolioOptimizer():
    """Predicts returns based on quintile profile of signal:
       works for one signal at a time.
       can be used to non-linearize signal.
    Attributes:
        num_buckets- 10.
        agg_type: mean/median.
    """
    def __init__(self, fac_data_dir, fac_ret_file, fac_cov_hl = 12, fac_names = None ):
        
        self.fac_data_dir = fac_data_dir
        self.fac_ret_file = fac_ret_file

        self.fac_ret = pd.read_csv(self.fac_ret_file, index_col = 'date')
        self.fac_ret.index = [ dt.datetime.strptime(val, '%Y-%m-%d') for val in self.fac_ret.index ]
        self.fac_ret = self.fac_ret.shift(1)
        
        fac_files = os.listdir(self.fac_data_dir)
        fac_files = np.sort(fac_files)
        
        
        fac_exp_df = pd.DataFrame()
        ret_sr = pd.Series()
        vol_sr = pd.Series()
        
    
        for file in fac_files :
            if '.csv' not in file :
                continue
            
            #print(file)              
            this_fac_df = pd.read_csv(self.fac_data_dir+'/'+file, index_col='date')
            this_fac_df.index = [ dt.datetime.strptime(val, '%Y-%m-%d') for val in this_fac_df.index ]
            
            ret_sr = ret_sr.append(this_fac_df['ret'])
            vol_sr = vol_sr.append(this_fac_df['vol'])
            this_fac_df.drop(['ret', 'vol'], inplace = True, axis=1)
            fac_exp_df = fac_exp_df.append(this_fac_df)
    
        self.rebalance_dates = (fac_exp_df.index.unique()).sort_values() 
        print('--------')
        
        if fac_names is not None :
            self.fac_exp_df = fac_exp_df[fac_names]
        
        self.fac_exp_df = fac_exp_df
        self.fac_cov_df = self.fac_ret.ewm(halflife=fac_cov_hl).cov()
        #vol is weekly . make it monthy 
        self.vol_sr = vol_sr*2
        self.ret_sr = ret_sr
        
    def run_signal(self,sig_sr, neu_sig = False ):
        #print('inside')
        sig_sr = sig_sr.copy()
        
        pnl_sr = []
        rebalance_dates = ( sig_sr.index.unique()).sort_values()        
        
        for this_date in rebalance_dates :
            #print(this_date)
            this_sig = sig_sr.loc[this_date]                                    
            this_fac_exp = self.fac_exp_df.loc[this_date]

            if neu_sig:
                lm_model = LinearRegression(fit_intercept = True)
                lm_model.fit(this_fac_exp,this_sig)
                res = this_sig - lm_model.predict(this_fac_exp)
                this_sig = res
                #this_sig = res.rank()
            else :
                temp = 'do nothing'
                #this_sig = this_sig.rank()
            #z_score of signal
            #this_sig = ( this_sig - this_sig.mean() )/ this_sig.std()  
            #this_sig = this_sig.fillna(0.0)
            
            this_fac_cov = self.fac_cov_df.loc[this_date]
            this_vol_sr = self.vol_sr.loc[this_date]
            this_vol_sr[this_vol_sr==0] = this_vol_sr[this_vol_sr !=0 ].mean()
            
            if neu_sig :
                this_cov_inv = np.diag(1/(this_vol_sr*this_vol_sr))
                #this_cov_inv = 1.0/(this_cov)
            else :    
                this_cov = np.dot(this_fac_exp,np.dot(this_fac_cov, this_fac_exp.T)) + np.diag(this_vol_sr*this_vol_sr)
                this_cov_inv = np.linalg.inv(this_cov)
            
            #this_cov_inv = np.linalg.inv(this_cov)
            weights = np.dot(this_cov_inv, this_sig)
            weights = weights / np.sum(  ( np.abs(weights) ) )
            
            this_ret = self.ret_sr.loc[this_date]
            this_pnl = np.dot(weights, this_ret)
            
            pnl_sr.append(this_pnl)
            
        pnl_sr = pd.Series(pnl_sr, index = rebalance_dates)
        print('-----------')
        return(pnl_sr)
    
    def neu_signal(self,sig_sr, fit_intercept ):
        sig_sr = sig_sr.copy()
        
        neu_sig_sr = pd.Series()
        rebalance_dates = ( sig_sr.index.unique()).sort_values()        
        
        for this_date in rebalance_dates :
            #print(this_date)
            this_sig = sig_sr.loc[this_date]                                    
            this_fac_exp = self.fac_exp_df.loc[this_date]

            
            lm_model = LinearRegression(fit_intercept = fit_intercept)
            lm_model.fit(this_fac_exp,this_sig)
            res = this_sig - lm_model.predict(this_fac_exp)
            this_sig = res
            #this_sig = res.rank(axis=0)

            #z_score of signal
            this_sig = ( this_sig - this_sig.mean() )/ this_sig.std()    
            this_sig = this_sig.fillna(0.0)
            neu_sig_sr = neu_sig_sr.append(this_sig)
            
        #neu_sig_sr = pd.Series(neu_sig_sr, index = rebalance_dates)
       
        return(neu_sig_sr)
        
       
            
