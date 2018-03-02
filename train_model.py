# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:01:56 2018

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import os as os

exec(open("./models/huber_model.py").read())

def get_model_names():
    model_names = [ 'linear_regression',
                    'knn',
                    'ridge_kernel',
                    'glmnet',
                    'huber_regression',
                    'pls_regression',
                    'decision_tree',
                    'gradient_boost',
                    'neural_net'
                    
                   ]
    return(model_names)

def get_default_model_arg(model_name):
    model_names = get_model_names()
    
    if model_name not in model_names :
        print(model_name + ' is not in the list')
        return {}
    
    if model_name == 'linear_regression':
        arg_dict ={'fit_intercept':True}
    
    if model_name == 'knn' :
        arg_dict = {'n_neighbors':10}
        
    if model_name == 'ridge_kernel':
        arg_dict = { 'kernel':'rbf','alpha':0.5,'gamma':0.3}
    
    if model_name == 'glmnet':
        arg_dict = { 'alpha':0.5, 'fit_intercept':True, 'n_lambda':1000,'tol':1e-8 }
        
    if model_name == 'huber_regression':
        arg_dict = {  'epsilon':1.35, 'max_iter':10000, 'alpha':0.00001,
                      'fit_intercept':True, 'tol':1e-07 }
        
    if model_name == 'pls_regression':
        arg_dict = { 'n_components':1, 'scale':False, 'max_iter':10000, 'tol':1e-08, 'copy':True }

#        class sklearn.tree.DecisionTreeRegressor
        
    if model_name == 'decision_tree':
       arg_dict = { 'criterion':'mse',  
                    'max_depth':None,  
                    'min_samples_leaf':10, 
                  }         
    
    if model_name == 'gradient_boost':
        arg_dict = { 'loss':'ls', 
                      'learning_rate':0.01, 
                      'n_estimators':100, 
                      'min_samples_leaf':10, 
                       'max_depth':10
                   }
    
    if model_name == 'neural_net':
        
        arg_dict = { 'hidden_layer_sizes':np.ones(5)*10, 
                   'activation':'relu', 
                   'solver':'lbfgs', 
                   'alpha':0.00001,  
                   }
    return arg_dict    
        

def train_model( X, y , model_name , model_arg = None,sample_weights = None, signs_vec = None):
    X = X.copy()
    y = y.copy()
    
    
    model_names = get_model_names()
    
    if model_name not in model_names :
        print(model_name + ' is not in the list')
        return None
    
    if model_arg is None:
        model_arg = get_default_model_arg(model_name)
    model_arg = model_arg.copy()

    
    model = None
    
    if model_name == 'linear_regression':
        model = LinearRegression(**model_arg)
        model.fit(X,y,sample_weight= sample_weights)
    
    if model_name == 'knn':
        model_arg['n_neighbors'] = int( X.shape[0]*1.0/model_arg['n_neighbors'] )
        model = KNeighborsRegressor(**model_arg)
        model.fit(X,y)
    
    if model_name == 'ridge_kernel':
        model = KernelRidge(**model_arg)
        model.fit(X,y,sample_weight= sample_weights)
    
    if model_name == 'glmnet':
        model = ElasticNet(**model_arg)
        model.fit(X,y,sample_weight= sample_weights, signs_vec=signs_vec)
        
    if model_name == 'huber_regression':
        model = HuberRegressor(**model_arg)
        model.fit(X,y,sample_weight= sample_weights, signs_vec=signs_vec)
    
    if model_name == 'pls_regression':
        #y = y.reshape( (y.shape[0], 1) )
        model = PLSRegression(**model_arg)
        model.fit(X,y)
    
    if model_name == 'decision_tree':
        model_arg['min_samples_leaf'] = int( X.shape[0]*1.0/model_arg['min_samples_leaf'] )
        model = DecisionTreeRegressor(**model_arg)
        model.fit(X,y,sample_weight= sample_weights)
    
    if model_name == 'gradient_boost':
        model_arg['min_samples_leaf'] = int( X.shape[0]*1.0/model_arg['min_samples_leaf'] )
        model = GradientBoostingRegressor(**model_arg)
        model.fit(X,y,sample_weight= sample_weights)
    
    if model_name == 'neural_net':
        model = MLPRegressor(**model_arg)
        model.fit(X,y)
        

    return ( model )

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

def predict_model(model_name, model, X, arg_dict ={} ):
    arg_dict = arg_dict.copy()
    
    arg_dict['X'] = X

    if model_name == 'glmnet':
       if 'lamb' in arg_dict.keys() :
           s, w_ind = get_lambda(model, arg_dict['lamb'] )
           arg_dict['lamb'] = s
           

    y = model.predict(**arg_dict)       

    if model_name == 'pls_regression':
        y = y[:,0]
        
    return (y)
      
    

      
        