"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import json
import sys
sys.path.append('..')

def readJson(file_name):
    return json.load(open(file_name))

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
