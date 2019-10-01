#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from models.utils import DataSet,readJson,LOG_INFO

import time
import os
import numpy as np 
config_data=readJson('config.json')

class train_args:
    data_dir    = config_data['train']['data_dir']
    save_dir    = config_data['train']['save_dir']
    image_dim   = config_data['train']['image_dim']
    dataset_name= config_data['train']['dataset_name']
    rename_flag = config_data['train']['rename_flag'] 
    tamper_iden = config_data['train']['tamper_iden']
    orig_iden   = config_data['train']['orig_iden']
    data_size   = config_data['train']['data_size'] 

class test_args:
    data_dir    = config_data['test']['data_dir']
    save_dir    = config_data['test']['save_dir']
    image_dim   = config_data['test']['image_dim']
    dataset_name= config_data['test']['dataset_name']
    rename_flag = config_data['test']['rename_flag'] 
    tamper_iden = config_data['test']['tamper_iden']
    orig_iden   = config_data['test']['orig_iden']
    data_size   = config_data['test']['data_size'] 

    
def main(argv):
    start_time=time.time()

    LOG_INFO('CREATING TRAINING DATA with {}'.format(train_args.dataset_name),p_color='yellow')
    obj=DataSet(train_args)
    obj.preprocess()
    LOG_INFO('CREATING TESTING DATA with {}'.format(test_args.dataset_name),p_color='yellow')
    obj=DataSet(test_args)
    obj.preprocess()
        
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
if __name__ == "__main__":
    main('MERUL')