# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:36:49 2018

@author: raghuramkowdeed
"""


import os as os
import shutil
import pandas as pd
import numpy as np

def create_dir(directory, del_dir = False):
    
    if not os.path.exists(directory):
       os.makedirs(directory)
    else :
       if del_dir :  
          shutil.rmtree(directory)
          os.makedirs(directory)
       