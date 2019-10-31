#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
#--------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Forged Image To Original Image Reconstruction',
                                formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("exec_flag", 
                    help='''
                            Execution Flag for creating files 
                            Available Flags: prep,train,eval,comb
                            png       = create images
                            tfrecords = create tfrecords
                            comb      = combined execution
                            PLEASE NOTE:
                            For Separate Run the following order must be maintained:
                            1) png
                            2) tfrecords
                            
                            ''')
args = parser.parse_args()
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
#-----------------------------------------------------------------------------------------------------------------------------------
def crop_len(nb_data,batch_size):
    return (nb_data//batch_size)*batch_size

def split_len(factor,nb_data):
    return  nb_data - int(factor*nb_data)    

def tfcreate(paths,DS,mode):
    if mode=='test':
        new_paths=paths
    else:
        new_paths=paths[:crop_len(len(paths),DS.STATICS.batch_size)]

    LOG_INFO('Creating TFRecords for {} Data'.format(mode))
    fs=DS.STATICS.file_size
    for i in range(0,len(new_paths),fs):
        image_paths= new_paths[i:i+fs]        
        r_num=i // fs
        to_tfrecord(image_paths,DS,mode,r_num)
#-----------------------------------------------------------------------------------------------------------------------------------

def create_trainData():
    DS=DataSet(ARGS.MICC_F2000,'train',ARGS.OUTPUT_DIR,STATICS)
    image_paths=glob(os.path.join(DS.image_dir,'*.png'))
    tfcreate(image_paths,DS,'train')

def create_testData():
    DS=DataSet(ARGS.MICC_F220,'test',ARGS.OUTPUT_DIR,STATICS)
    image_paths=glob(os.path.join(DS.image_dir,'*.png'))
    tfcreate(image_paths,DS,'test')
#-----------------------------------------------------------------------------------------------------------------------------------

def main(args):
    start_time=time.time()
    if args.exec_flag=='png':
        create_png()
    elif args.exec_flag=='tfrecords':
        create_trainData()
        create_testData()
    elif args.exec_flag=='comb':
        create_png()
        create_trainData()
        create_testData()
    else:
        raise ValueError('CHECK FLAG: (png,tfrecords,comb)')
            
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main(args)