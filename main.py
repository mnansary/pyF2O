#!/usr/bin/env python3
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from core.utils import LOG_INFO
from scripts.png import create_png
from scripts.trainH5 import create_trainH5
from scripts.testH5 import create_testH5
from scripts.tfrecords import create_tfrecord

import time
import os
import numpy as np 
import json
import platform

from subprocess import Popen, PIPE


def readJson(file_name):
    return json.load(open(file_name))

def clear_cache():
    proc = Popen(['free', '-m'], stdout=PIPE)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print (colored(line.rstrip(),'red'))

    os.system('sudo {}'.format(os.path.join(os.getcwd(),'clear_cache.sh')))          
        
    proc = Popen(['free', '-m'], stdout=PIPE)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print (colored(line.rstrip(),'green'))

def main():
    start_time=time.time()
    CLEAR_CACHE=True

    if platform.system()!='Linux':
        LOG_INFO('RAM/BUFFER/CACHE AUTO CLEAR NOT IMPLEMENTED FOR {}'.format(platform.system()),p_color='red')
        LOG_INFO('To avoid error/ crash or if faced with error, execute the scripts in scripts folder separately for safe operation.',p_color='green')
        CLEAR_CACHE=False
    
    create_png()
    
    if CLEAR_CACHE:
        clear_cache()

    create_trainH5()
    
    if CLEAR_CACHE:
        clear_cache()
    
    create_testH5()
    
    if CLEAR_CACHE:
        clear_cache()
    
    create_tfrecord()
    
    if CLEAR_CACHE:
        clear_cache()
        
    LOG_INFO('Time Taken:{} s'.format(time.time()-start_time),p_color='yellow')
    
    
if __name__ == "__main__":
    main()