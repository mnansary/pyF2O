#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

# TODO: Creating Batch wise tfrecord (Future)

from models.utils import DataSet,readJson,readh5,LOG_INFO,to_tfrecord,create_h5_dataset

import time
import h5py
import os

config_data=readJson('config.json')

class train_args:
    data_dir    = config_data['train']['data_dir']
    save_dir    = config_data['train']['save_dir']
    image_dim   = config_data['train']['image_dim']
    dataset_name= config_data['train']['dataset_name']
    rename_flag = config_data['train']['rename_flag'] 
    tamper_iden = config_data['train']['tamper_iden']
    orig_iden   = config_data['train']['orig_iden'] 

class test_args:
    data_dir    = config_data['test']['data_dir']
    save_dir    = config_data['test']['save_dir']
    image_dim   = config_data['test']['image_dim']
    dataset_name= config_data['test']['dataset_name']
    rename_flag = config_data['test']['rename_flag'] 
    tamper_iden = config_data['test']['tamper_iden']
    orig_iden   = config_data['test']['orig_iden'] 

class PARAMS:
    save_h5          = config_data['PARAMS']['save_h5']
    save_tfrecord    = config_data['PARAMS']['save_tfrecord']
    create_dataset   = config_data['PARAMS']['create_dataset']
    

def main(argv):
    start_time=time.time()

    if PARAMS.create_dataset==1:
        LOG_INFO('CREATING TRAINING DATA with {}'.format(train_args.dataset_name),p_color='yellow')
        obj=DataSet(train_args)
        obj.preprocess()
        LOG_INFO('CREATING TESTING DATA with {}'.format(test_args.dataset_name),p_color='yellow')
        obj=DataSet(test_args)
        obj.preprocess()
    
    if PARAMS.save_h5==1:
        LOG_INFO('H5 TRAINING DATA with {}'.format(train_args.dataset_name),p_color='yellow')
        create_h5_dataset(train_args,'Train')
        LOG_INFO('H5 TESTING DATA with {}'.format(test_args.dataset_name),p_color='yellow')
        create_h5_dataset(test_args,'Test')
        
        X_Train_path=os.path.join(obj.h5dir,'X_Train.h5')
        Y_Train_path=os.path.join(obj.h5dir,'Y_Train.h5')
        X_Test_path=os.path.join(obj.h5dir,'X_Test.h5')
        Y_Test_path=os.path.join(obj.h5dir,'Y_Test.h5')

        X_Train=readh5(X_Train_path)
        Y_Train=readh5(Y_Train_path)
        X_Test=readh5(X_Test_path)
        Y_Test=readh5(Y_Test_path)

        LOG_INFO('Train-Images-Tensor: {}'.format(X_Train.shape))
        LOG_INFO('Train-Target-Tensor: {}'.format(Y_Train.shape))
        LOG_INFO('Test-Images-Tensor: {}'.format(X_Test.shape))
        LOG_INFO('Test-Target-Tensor: {}'.format(Y_Test.shape))

    if PARAMS.save_tfrecord==1:    
        LOG_INFO('CREATING tfrecords',p_color='yellow')
        to_tfrecord(train_args,'Train')
        to_tfrecord(test_args,'Test')

    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
if __name__ == "__main__":
    main('MERUL')