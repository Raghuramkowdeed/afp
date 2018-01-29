#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:57:00 2018

@author: raghuramkowdeed
"""

import pandas as pd
import numpy as np

class QuintileSort():
    """Predicts returns based on quintile profile of signal:
       works for one signal at a time.
       can be used to non-linearize signal.
    Attributes:
        num_buckets- 10.
        agg_type: mean/median.
    """

    def __init__(self, num_buckets, agg_type):
        
        self.num_buckets = num_buckets
        self.agg_type = agg_type
        self.ret_profile = None

    def fit(self, x, y):
       '''
       x, y are Series objects.
       make sure both x,y have same index, size
       '''
       x = x.copy()
       y = y.copy()
       
       y_sd = y.std()
       y[y>3*y_sd] = 3*y_sd
       y[y<-3*y_sd] = -3*y_sd
       
       
       X = pd.DataFrame()
       X['sig'] = x
       X['ret'] = y
       X['rank'] = X['sig'].rank()
       X = X.sort_values(['rank'])
       min_rank = X['rank'].min()
       max_rank = X['rank'].max()
       X['rank'] = ( X['rank'] - min_rank ) / (max_rank - min_rank)
       
       num_buckets = self.num_buckets

       dis_ranks = pd.Series( np.zeros(X.shape[0]), index = X.index ) * np.nan

       for i in range(num_buckets):
           low_thrs = float(i)/float( num_buckets )
           high_thrs = float(i+1)/float( num_buckets )
           ind = (X['rank']<high_thrs ) & (X['rank']>=low_thrs)
           dis_ranks[ind] = i    
           
       X['dis_rank'] = dis_ranks
       X = X[ ['dis_rank', 'ret'] ]
       X = X.groupby(['dis_rank']).aggregate(self.agg_type)
      
       self.ret_profile = X.copy()
       self.ret_profile = self.ret_profile.iloc[:,0]
       del X

    def predict(self, x):
        """
        x is Series object, 
        returns predcits ret Series obj
        """
        x = x.copy()
        X = pd.DataFrame()
        X['sig'] = x
        X['rank'] = x.rank()
        X = X.sort_values(['rank'])
        min_rank = X['rank'].min()
        max_rank = X['rank'].max()
        X['rank'] = ( X['rank'] - min_rank ) / (max_rank - min_rank)

        
        num_buckets = self.num_buckets

        dis_ranks = pd.Series( np.zeros(X.shape[0]), index = X.index ) * np.nan

        for i in range(num_buckets):
           low_thrs = float(i)/float( num_buckets )
           high_thrs = float(i+1)/float( num_buckets )
           ind = (X['rank']<high_thrs ) & (X['rank']>=low_thrs)
           dis_ranks[ind] = i    
       
        X['dis_rank'] = dis_ranks
        
        pred_ret = self.ret_profile.loc[X['dis_rank']]
        pred_ret.index = X.index
        
        del X
        
        return pred_ret