"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from F2O.utils import data_input_fn

import numpy as np 
import matplotlib.pyplot as plt 


import tensorflow as tf 

tf.compat.v1.enable_eager_execution()

class FLAGS:
    TFRECORDS_DIR  = '/home/ansary/RESEARCH/F2O/Data/ORG/DataSet/tfrecord/'
    MODE            = 'train'
    IMAGE_DIM       = 256
    NB_CHANNELS     = 3
    BATCH_SIZE      = 128
    SHUFFLE_BUFFER  = 672
    DEBUG           = True
   
NB_TOTAL_DATA       = 672 
    
def check_data():
    dataset=data_input_fn(FLAGS)
    COUNT=10
    counter=0
    for imgs,gts in dataset:
        print(imgs.shape,gts.shape)
        for ii in range(imgs.shape[0]):
            counter+=1
            dat=np.concatenate((imgs[ii],gts[ii]),axis=1)
            plt.imshow(dat)
            plt.show()
            if counter>=COUNT:
                break
        if counter>=COUNT:
                break
                
    
#--------------------------------------------------------------------------------------------------
N_EPOCHS            = 2
STEPS_PER_EPOCH     =  NB_TOTAL_DATA //FLAGS.BATCH_SIZE 

def train_in_fn():
    return data_input_fn(FLAGS)

def build():
    IN=tf.keras.layers.Input(shape=(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM, FLAGS.NB_CHANNELS))
    x = tf.keras.layers.Conv2D(64,(3,3), padding='same',strides=(1,1))(IN)
    x = tf.keras.layers.Conv2D(3,(3,3), padding='same',strides=(1,1))(x)
    return tf.keras.models.Model(inputs=IN,outputs=x)

def tarin_debug():
    model = build()
    model.summary()
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=tf.keras.losses.mean_absolute_error,
    )
    
    model.fit(
        train_in_fn(),
        epochs= N_EPOCHS,
        steps_per_epoch= STEPS_PER_EPOCH
    )
#--------------------------------------------------------------------------------------------------

#tarin_debug()
check_data()
