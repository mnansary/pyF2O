# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input,Concatenate,Reshape,Conv2D,UpSampling2D,LeakyReLU,BatchNormalization,MaxPooling2D,Activation,Flatten,Dense,Lambda
import tensorflow.keras.backend as K
import numpy as np
from keras.utils import plot_model
#-----------------------------------------------------------------------------------------------------------------------------
def unet(image_dim=128,nb_channels=3,kernel_size=(3,3),strides=(2,2),padding='same',alpha=0.2):
    # input 
    in_image_shape=(image_dim,image_dim,nb_channels)
    # U-Net
    min_nb_filter=64
    max_nb_filter=512
    nb_conv_layers = int(np.floor(np.log(image_dim) / np.log(2)))

    nb_filters = [min_nb_filter * min((max_nb_filter/min_nb_filter) , (2 ** i)) for i in range(nb_conv_layers)]
    nb_filters = list(map(int, nb_filters))

    ## Encoder
    en_X=[]
    # Input Block
    X=Input(shape=in_image_shape,name='gen_enc_init_input')
    In=X

    # Conv Blocks
    for index,nb_filter in enumerate(nb_filters):
        if index>0:
            X=LeakyReLU(alpha=alpha,name='gen_enc_conv_{}_act'.format(index+1))(X)
        X=Conv2D(nb_filter,kernel_size,name='gen_enc_conv_{}'.format(index+1),strides=strides,padding=padding)(X)
        if index > 0:
            X=BatchNormalization(name='gen_enc_conv_{}_bn'.format(index+1))(X)
        en_X.append(X)


    # Decoder filters
    nb_filters = nb_filters[:-2][::-1] 
    if len(nb_filters) < nb_conv_layers - 1:
        nb_filters.append(min_nb_filter)

    # UpSampling Blocks
    for index,nb_filter in enumerate(nb_filters):
        X = Activation("relu",name='gen_dec_conv_{}_act'.format(index+1))(X)
        X = UpSampling2D(size=(2, 2),name='gen_dec_conv_{}_ups'.format(index+1))(X)
        X = Conv2D(nb_filter, kernel_size, name='gen_dec_conv_{}'.format(index+1), padding="same")(X)
        X = BatchNormalization(name='gen_dec_conv_{}_bn'.format(index+1))(X)
        X = Concatenate(name='gen_dec_conv_{}_conc'.format(index+1))([X ,en_X[-(index+2)]])

    X = Activation("relu",name='gen_dec_conv_last_act')(X)
    X = UpSampling2D(size=(2, 2),name='gen_dec_conv_last_ups')(X)
    X = Conv2D(nb_channels, kernel_size, name="gen_dec_conv_last", padding="same")(X)
    X = Activation("tanh",name='gen_dec_conv_final_act')(X)

    gen_unet_model = Model(inputs=[In], outputs=[X])

    return gen_unet_model

if __name__ == "__main__":
    model=unet()
    model.summary()
    plot_model(model,to_file='UNET.png',show_shapes=True)