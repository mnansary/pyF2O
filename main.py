#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from core.utils import LOG_INFO
from scripts.png import create_png
from scripts.H5s import create_trainH5,create_testH5
from scripts.tfrecords import create_tfrecord

import time
import os
import numpy as np 
import json


def readJson(file_name):
    return json.load(open(file_name))


def main():
    start_time=time.time()
    create_png()
    create_trainH5()
    create_testH5()
    create_tfrecord()    
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main()