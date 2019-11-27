# -*- coding: utf-8 -*-
from __future__ import print_function
from termcolor import colored

import tensorflow as tf
import numpy as np 
import os
# From official Tensorflow tutorial: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
#---------------------------------------------------------------------------------------------------------------------------
def upsample(filters, size,idx,apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name='UPSAMPLE_{}'.format(idx))
    result.add(
               tf.keras.layers.Conv2DTranspose(filters, 
                                               size, 
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False
                                               )
               ) 

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def downsample(filters, size,idx,apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name='DOWNSAMPLE_{}'.format(idx))
    result.add(
            tf.keras.layers.Conv2D(filters, 
                                    size, 
                                    strides=2, 
                                    padding='same',
                                    kernel_initializer=initializer, 
                                    use_bias=False)
            )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result
#---------------------------------------------------------------------------------------------------------------------------
def Generator(IMAGE_DIM=256,NB_CHANNELS=3):
    inputs = tf.keras.layers.Input(shape=[IMAGE_DIM,IMAGE_DIM,NB_CHANNELS])

    down_stack = [
        downsample(64, 4,1, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4,2), # (bs, 64, 64, 128)
        downsample(256, 4,3), # (bs, 32, 32, 256)
        downsample(512, 4,4), # (bs, 16, 16, 512)
        downsample(512, 4,5), # (bs, 8, 8, 512)
        downsample(512, 4,6), # (bs, 4, 4, 512)
        downsample(512, 4,7), # (bs, 2, 2, 512)
        downsample(512, 4,8), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4,1, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4,2, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4,3, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4,4), # (bs, 16, 16, 1024)
        upsample(256, 4,5), # (bs, 32, 32, 512)
        upsample(128, 4,6), # (bs, 64, 64, 256)
        upsample(64, 4,7), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(NB_CHANNELS, 
                                            4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
#---------------------------------------------------------------------------------------------------------------------------
def Discriminator(IMAGE_DIM=256,NB_CHANNELS=3):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, NB_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMAGE_DIM, IMAGE_DIM, NB_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4,'DISC_1', False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4,'DISC_2')(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4,'DISC_3')(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 
                                  4, 
                                  strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 
                                  4, 
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
#---------------------------------------------------------------------------------------------------------------------------
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#---------------------------------------------------------------------------------------------------------------------------
def generator_loss(disc_generated_output, gen_output, target,LAMBDA=100):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
#---------------------------------------------------------------------------------------------------------------------------
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
#---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    img_path='/home/ansary/RESEARCH/F2O/pyF2O/INFO/'
    gen=Generator()
    tf.keras.utils.plot_model(gen,to_file=os.path.join(img_path,'gen.png'),show_layer_names=True,show_shapes=True)
    dis=Discriminator()
    tf.keras.utils.plot_model(dis,to_file=os.path.join(img_path,'dis.png'),show_layer_names=True,show_shapes=True)
    