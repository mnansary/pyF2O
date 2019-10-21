"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import time
import os
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt 

import sys
sys.path.append('..')


from F2O.generators import unet,man_net
from F2O.utils import plot_data,LOG_INFO

import imageio
from PIL import Image as imgop
import random


import argparse
parser = argparse.ArgumentParser(description='prediction Script')
parser.add_argument("model_dir", help="/path/to/model/weights/h5file")
parser.add_argument("test_dir",help="/path/to/test/folder")
args = parser.parse_args()



def main(args,viz=False):
    
    # Detect Model
    if 'manNet' in args.model_dir:
        model_name='manNet'
    else:
        model_name='uNet'

    LOG_INFO('Loading Model')
    if model_name=='uNet':
        model=unet()
    elif model_name=='manNet':
        model=man_net()
    else:
        raise ValueError('NOT Implemented !! Use uNet or manNet as model name')
    
    model.load_weights(args.model_dir)
    
    LOG_INFO('Generating Predictions: {}'.format(model_name))

    img_dir=os.path.join(args.test_dir,'image')
    img_files=glob(os.path.join(img_dir,'*.png'))
    
    for img_path in img_files:
        # get gt path
        gt_path=str(img_path).replace('image','target')
        # read img and gt
        img=imgop.open(img_path)
        gt=imgop.open(gt_path)
        #normalize
        img=np.array(img)
        img=img.astype('float32')/255.0
        # Ground Truth
        gt=np.array(gt)
        gt=gt.astype('float32')/255.0 
        #tensor
        tensor=np.expand_dims(img,axis=0)
        # prediction
        pred=np.squeeze(model.predict(tensor))
        # concat
        data=np.concatenate((img,gt,pred),axis=1)
        plt.imshow(data)
        plt.show()
        plt.clf()
        plt.close()
        
            




if __name__=='__main__':
    main(args,viz=True)
