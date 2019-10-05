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

class ARGS:
    MICC_F2000     = config_data['ARGS']['MICC-F2000']
    MICC_F220      = config_data['ARGS']['MICC-F220']
    OUTPUT_DIR     = config_data['ARGS']['OUTPUT_DIR']

class STATICS:
    tamper_iden     = config_data['STATICS']["tamper_iden"]          
    orig_iden       = config_data['STATICS']["orig_iden"]             
    image_dim       = config_data['STATICS']["image_dim"]    
    nb_channels     = config_data['STATICS']["nb_channels"]            
    rot_angle_start = config_data['STATICS']["rot_angle_start"]      
    rot_angle_end   = config_data['STATICS']["rot_angle_end"]    
    rot_angle_step  = config_data['STATICS']["rot_angle_step"]    
    shade_factor    = config_data['STATICS']["shade_factor"]    
    train_eval_split= config_data['STATICS']["train_eval_split"]  
    file_size       = config_data['STATICS']["file_size"]   
    batch_size      = config_data['STATICS']["batch_size"]   
    fid_num         = config_data['STATICS']["fid_num"]  
    rename_data     = config_data['STATICS']["rename_data"]   
    prob_idens      = config_data['STATICS']["prob_idens"]   
#-----------------------------------------------------------------------------------------------------------------------------------

import time
import os
import numpy as np 
from glob import glob
from F2O.utils import LOG_INFO,DataSet,to_tfrecord
#-----------------------------------------------------------------------------------------------------------------------------------

def create_png():
    start_time=time.time()
    LOG_INFO('CREATING TRAINING DATA' ,p_color='yellow')
    TRAIN_DS=DataSet(ARGS.MICC_F2000,'train',ARGS.OUTPUT_DIR,STATICS)
    TRAIN_DS.create()
    LOG_INFO('CREATING TESTING DATA ',p_color='yellow')
    TEST_DS=DataSet(ARGS.MICC_F220,'test',ARGS.OUTPUT_DIR,STATICS)
    TEST_DS.create()
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    return TRAIN_DS
#-----------------------------------------------------------------------------------------------------------------------------------
def crop_len(nb_data,batch_size):
    return (nb_data//batch_size)*batch_size

def split_len(factor,nb_data):
    return  nb_data - int(factor*nb_data)    

def tfcreate(paths,DS,mode):
    new_paths=paths[:crop_len(len(paths),DS.STATICS.batch_size)]
    LOG_INFO('Creating TFRecords for {} Data'.format(mode))
    fs=DS.STATICS.file_size
    for i in range(0,len(new_paths),fs):
        image_paths= new_paths[i:i+fs]        
        r_num=i // fs
        to_tfrecord(image_paths,DS,mode,r_num)
    


def create_tfrecord(DS):
    start_time=time.time()
    image_path_list=glob(os.path.join(DS.image_dir,'*.png'))
    # eval and train split
    _len=split_len(DS.STATICS.train_eval_split,len(image_path_list)) 
    train_image_paths=image_path_list[:_len]
    eval_image_paths=image_path_list[_len:]
    # tfrecords
    tfcreate(train_image_paths,DS,'train')
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    tfcreate(eval_image_paths ,DS,'eval' )
#-----------------------------------------------------------------------------------------------------------------------------------

def main(arg):
    start_time=time.time()
    TRAIN_DS=create_png()

    create_tfrecord(TRAIN_DS)    
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main('MERUL')