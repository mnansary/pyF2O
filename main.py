#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

#-----------------------------------------------------Load Config----------------------------------------------------------
import json

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
    batch_size  = config_data['train']['batch_size'] 

class test_args:
    data_dir    = config_data['test']['data_dir']
    save_dir    = config_data['test']['save_dir']
    image_dim   = config_data['test']['image_dim']
    dataset_name= config_data['test']['dataset_name']
    rename_flag = config_data['test']['rename_flag'] 
    tamper_iden = config_data['test']['tamper_iden']
    orig_iden   = config_data['test']['orig_iden']
    batch_size  = config_data['test']['batch_size'] 

#-----------------------------------------------------------------------------------------------------------------------------------

import time
import os
import numpy as np 
from glob import glob
from F2O.utils import LOG_INFO,DataSet,to_tfrecord
#-----------------------------------------------------------------------------------------------------------------------------------

def create_png():
    start_time=time.time()
    LOG_INFO('CREATING TRAINING DATA with {}'.format(train_args.dataset_name),p_color='yellow')
    obj=DataSet(train_args)
    obj.preprocess()
    LOG_INFO('CREATING TESTING DATA with {}'.format(test_args.dataset_name),p_color='yellow')
    obj=DataSet(test_args)
    obj.preprocess()
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
#-----------------------------------------------------------------------------------------------------------------------------------
def create_tfrecord(args):
    start_time=time.time()
    obj=DataSet(args)
    # get all paths
    image_path_list=glob(os.path.join(obj.save_dir,'*.png'))
    # exclude Target Paths
    image_path_list=[img_path for img_path in image_path_list if "_target" not in img_path]  
    # crop to batced length
    bs=int(args.batch_size)
    nb_data =len(image_path_list)
    crop_len=(nb_data//bs)*bs
    image_path_list=image_path_list[:crop_len]
    nb_data =len(image_path_list)
    # eval and train split (0.2)
    nb_files=nb_data//bs
    eval_nb=int(nb_files* 0.2)
    train_nb=nb_files - eval_nb

    train_len=train_nb*bs 
    
    train_image_paths=image_path_list[:train_len]
    eval_image_paths=image_path_list[train_len:]

    # train files
    obj.mode='Train'
    LOG_INFO('Creating TFRecords for {} Data'.format(obj.mode))
    for i in range(0,len(train_image_paths),bs):
        image_paths= train_image_paths[i:i+bs]        
        r_num=i // bs
        to_tfrecord(image_paths,obj,r_num)
    
    # eval files
    obj.mode='Eval'
    LOG_INFO('Creating TFRecords for {} Data'.format(obj.mode))
    for i in range(0,len(eval_image_paths),bs):
        image_paths= eval_image_paths[i:i+bs]        
        r_num=i // bs
        to_tfrecord(image_paths,obj,r_num)
    
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
#-----------------------------------------------------------------------------------------------------------------------------------

def main(arg):
    start_time=time.time()
    create_png()
    create_tfrecord(train_args)    
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main('MERUL')