# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import tensorflow as tf
import numpy as np 
# this code is simply copied from: https://github.com/agermanidis/pix2pix-tpu/blob/master/model.py with some minor changes
#---------------------------------------------------------------------------------------------------------------------------
initializer = tf.random_normal_initializer(0, 0.02)    
EPS = 1e-12
#---------------------------------------------------------------------------------------------------------------------------
def _leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)
def _batch_norm(x):
    return tf.keras.layers.BatchNormalization(gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(x)
def _dense(x, channels):
    return tf.keras.layers.Dense(channels)(x)
def _conv2d(x, filters, kernel_size=3, stride=2):
    return tf.keras.layers.Conv2D(filters,kernel_size,strides=(stride,stride),padding='same',kernel_initializer=initializer)(x)
def _deconv2d(x, filters, kernel_size=3, stride=2):
    return tf.keras.layers.Conv2DTranspose(filters,kernel_size,strides=(stride,stride),padding='same',kernel_initializer=initializer)(x)
#---------------------------------------------------------------------------------------------------------------------------
def generator_fn(generator_inputs):
    # fixed args
    generator_outputs_channels    =   3
    ngf  =   64 
    # encoder layer specs
    layer_specs = [
        # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 2,
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 4,
        # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,
        # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,
        # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,
        # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,
        # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ngf * 8,
    ]
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.compat.v1.variable_scope("encoder_1"):
        output = _conv2d(generator_inputs, ngf)
        layers.append(output)
    
    for out_channels in layer_specs:
        with tf.compat.v1.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = _leaky_relu(layers[-1])
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = _conv2d(rectified, out_channels)
            output = _batch_norm(convolved)
            layers.append(output)
    # decoder layers 
    layer_specs = [
        # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),
        # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 4, 0.0),
        # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf * 2, 0.0),
        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        (ngf, 0.0),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.compat.v1.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                _input = layers[-1]
            else:
                _input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(_input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = _deconv2d(rectified, out_channels)
            output = _batch_norm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.compat.v1.variable_scope("decoder_1"):
        _input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(_input)
        output = _deconv2d(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]
#---------------------------------------------------------------------------------------------------------------------------
def discriminator_fn(discrim_inputs, discrim_targets):
    n_layers = 3
    ndf =   64
    layers = []
    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    _input = tf.concat([discrim_inputs, discrim_targets], axis=3)
    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.compat.v1.variable_scope("layer_1"):
        convolved = _conv2d(_input,ndf)
        rectified = _leaky_relu(convolved)
        layers.append(rectified)
    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.compat.v1.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = _conv2d(layers[-1], out_channels, stride=stride)
            normalized = _batch_norm(convolved)
            rectified = _leaky_relu(normalized)
            layers.append(rectified)
    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.compat.v1.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = _conv2d(rectified,1,stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)
    return layers[-1]
#---------------------------------------------------------------------------------------------------------------------------
def loss_fn(inputs, targets):
    gan_weight  =   100.0
    l1_weight   =   1.0
    # --> Generated output
    with tf.compat.v1.variable_scope("generator"):
        outputs = generator_fn(inputs)
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    # --> Real Predictions
    with tf.compat.v1.name_scope("real_discriminator"):
        with tf.compat.v1.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = discriminator_fn(inputs, targets)
    # --> Fake Predictions
    with tf.compat.v1.name_scope("fake_discriminator"):
        with tf.compat.v1.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = discriminator_fn(inputs, outputs)
    # --> Disc Loss
    with tf.compat.v1.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    # --> Gen Loss
    with tf.compat.v1.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
    return gen_loss + discrim_loss


    