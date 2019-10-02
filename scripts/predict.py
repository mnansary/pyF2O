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

import argparse

parser = argparse.ArgumentParser(description='prediction Script')
parser.add_argument("model_dir", help="/path/to/model/weights/h5file")

args = parser.parse_args()

from core.generators import unet,man_net
from core.utils import plot_data,readh5,LOG_INFO
from scripts.config import test_args
from core.utils import DataSet
import imageio
import random

def main(args,viz=False):
    # Load Data
    LOG_INFO('Loading Test Data')
    obj=DataSet(test_args)
    X_dir=os.path.join(obj.base_save_dir,'H5Data','X_Test_ALL.h5')
    Y_dir=os.path.join(obj.base_save_dir,'H5Data','Y_Test_ALL.h5')
    imgs=readh5(X_dir)
    gts=readh5(Y_dir)

    # Detect Model
    if 'manNet' in args.model_dir:
        model_name='manNet'
    else:
        model_name='uNet'

    save_dir=os.path.join(obj.base_save_dir,model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    LOG_INFO('Loading Model')
    if model_name=='uNet':
        model=unet()
    elif model_name=='manNet':
        model=man_net()
    else:
        raise ValueError('NOT Implemented !! Use uNet or manNet as model name')
    
    model.load_weights(args.model_dir)
    
    LOG_INFO('Generating Predictions: {}'.format(model_name))
    for idx,img in enumerate(imgs):
        #normalize
        arr=img.astype('float32')/255.0
        #tensor
        tensor=np.expand_dims(arr,axis=0)
        #save file path
        file_name=os.path.join(save_dir,'{}_{}_{}_prediction.png'.format(idx,random.randint(2000,5000),model_name))
        # prediction
        pred=np.squeeze(model.predict(tensor)) 
        # Ground Truth
        gt=np.squeeze(gts[idx])
        gt=gt.astype('float32')/255.0 
        # concat
        data=np.concatenate((arr,gt,pred),axis=1)
        # save img
        imageio.imsave(file_name,data)
        LOG_INFO('Saving: {}'.format(file_name))

        if viz:
            if (idx<10):
                plot_data(arr,gt,pred,model_name,save_flag=os.path.join(save_dir,'{}_info_{}.png'.format(idx,model_name)))
                LOG_INFO('Info: {}'.format(file_name))

if __name__=='__main__':
    main(args,viz=True)