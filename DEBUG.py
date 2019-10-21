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
    TFRECORDS_DIR  = '/home/ansary/RESEARCH/F2O/DataSet/tfrecord/'
    MODE            = 'train'
    IMAGE_DIM       = 256
    NB_CHANNELS     = 3
    BATCH_SIZE      = 4
    SHUFFLE_BUFFER  = 10
   
NB_TOTAL_DATA       = 48 
NB_EVAL_DATA        = 8
NB_TRAIN_DATA       = NB_TOTAL_DATA -  NB_EVAL_DATA 

    
def check_data():
    dataset=data_input_fn(FLAGS,'NOT NEEDED')
    for imgs,gts in dataset:
        print(imgs.shape,gts.shape)
        for ii in range(imgs.shape[0]):
            dat=np.concatenate((imgs[ii],gts[ii]),axis=1)
            plt.imshow(dat)
            plt.show()
    
#--------------------------------------------------------------------------------------------------
N_EPOCHS            = 2
STEPS_PER_EPOCH     =  NB_TOTAL_DATA //FLAGS.BATCH_SIZE 
VALIDATION_STEPS    =  NB_EVAL_DATA //FLAGS.BATCH_SIZE 

def train_in_fn():
    return data_input_fn(FLAGS,'train')
def eval_in_fn():
    FLAGS.MODE='eval'
    return data_input_fn(FLAGS,'eval')

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
        steps_per_epoch= STEPS_PER_EPOCH,
        validation_data=eval_in_fn(),
        validation_steps= VALIDATION_STEPS
    )
#--------------------------------------------------------------------------------------------------

tarin_debug()
check_data()
