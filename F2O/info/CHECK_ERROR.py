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
    TFRECORDS_DIR   = '/home/ansary/RESEARCH/F2O/DataSet/tfrecord/'
    IMAGE_DIM       = 256
    NB_CHANNELS     = 3
    BATCH_SIZE      = 8
    SHUFFLE_BUFFER  = 10000

NB_TOTAL_DATA       = 96
NB_EVAL_DATA        = 16
N_EPOCHS            = 2

NB_TRAIN    = ( NB_TOTAL_DATA -  NB_EVAL_DATA)// FLAGS.BATCH_SIZE
NB_EVAL     =  NB_EVAL_DATA // FLAGS.BATCH_SIZE

STEPS_PER_EPOCH     = int( NB_TOTAL_DATA //FLAGS.BATCH_SIZE )
VALIDATION_STEPS    = int( NB_EVAL_DATA //FLAGS.BATCH_SIZE )

    
#---------------------------------------------------------------------------------------------

def train_in_fn():
    return data_input_fn(FLAGS,'Train')

def eval_in_fn():
    return data_input_fn(FLAGS,'Eval')



#---------------------------------------------------------------------------------------------
def check_data(dataset,NB_DATA):
    for i in range(NB_DATA):
        for imgs,gts in dataset:
            print('BatchNum:{}'.format(i+1))
            print(imgs.shape, gts.shape)
            for ii in range(imgs.shape[0]):
                data=np.concatenate((imgs[ii],gts[ii]),axis=1)
                plt.imshow(data)
                plt.show()
                plt.clf()
                plt.close()
            
#--------------------------------------------------------------------------------------------------
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
    
#check_data(train_in_fn(),NB_TRAIN)
#check_data(eval_in_fn(),NB_EVAL)
tarin_debug()