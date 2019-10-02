"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
import sys
sys.path.append("..")

from scripts.config import train_args,test_args
from core.utils import DataSet,LOG_INFO


def create_png():
    start_time=time.time()

    LOG_INFO('CREATING TRAINING DATA with {}'.format(train_args.dataset_name),p_color='yellow')
    obj=DataSet(train_args)
    obj.preprocess()
    LOG_INFO('CREATING TESTING DATA with {}'.format(test_args.dataset_name),p_color='yellow')
    obj=DataSet(test_args)
    obj.preprocess()
        
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
if __name__ == "__main__":
    create_png()