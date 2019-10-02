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
from PIL import Image as imgop

import sys
sys.path.append("..") 

from core.utils import DataSet,LOG_INFO,readh5,H5_data
from scripts.config import train_args,test_args



def create_trainH5():
    start_time=time.time()

    LOG_INFO('Creating H5 for Train Data')
    
    obj=DataSet(train_args)
    image_path_list=glob(os.path.join(obj.save_dir,'*.png'))
    H5_data(image_path_list,obj,'ALL')
    LOG_INFO('Total Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')

def create_testH5():
    start_time=time.time()

    LOG_INFO('Creating H5 for Test Data')
    
    obj=DataSet(test_args)
    image_path_list=glob(os.path.join(obj.save_dir,'*.png'))
    H5_data(image_path_list,obj,'ALL')
    LOG_INFO('Total Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')


if __name__=='__main__':
    create_trainH5()
    create_testH5()
    