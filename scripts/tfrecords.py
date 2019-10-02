#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
from glob import glob

import sys 
sys.path.append("..")

from core.utils import to_tfrecord,LOG_INFO,DataSet
from scripts.config import train_args,test_args


def path_to_record(args):
    obj=DataSet(args)
    image_path_list=glob(os.path.join(obj.save_dir,'*.png'))
    for i in range(0,len(image_path_list),int(args.data_size)):
        image_paths= image_path_list[i:i+int(args.data_size)]
        if i==0:
            r_num=0
        else:
            r_num=int(i // int(args.data_size))
        to_tfrecord(image_paths,obj,r_num)
    

def create_tfrecord():
    start_time=time.time()
    LOG_INFO('Creating TFRecords for Test Data')
    path_to_record(test_args)
    LOG_INFO('Creating TFRecords for Train Data')
    path_to_record(train_args)
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')

if __name__=='__main__':
    create_tfrecord()
    