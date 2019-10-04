# -*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input,Concatenate,Reshape,Conv2D,Conv2DTranspose,LeakyReLU,BatchNormalization,MaxPooling2D,Activation,Flatten,Dense,Lambda
import tensorflow.keras.backend as K

import numpy as np
import tensorflow as tf 
import os 


from tensorflow.keras.utils import plot_model
#-----------------------------------------------------------------------------------------------------------------------------
def unet(image_dim=256,nb_channels=3,kernel_size=(3,3),strides=(2,2),padding='same',alpha=0.2):
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
        X = Conv2DTranspose(nb_filter, kernel_size,strides=strides,name='gen_dec_deconv_{}'.format(index+1), padding="same")(X)
        X = BatchNormalization(name='gen_dec_deconv_{}_bn'.format(index+1))(X)
        X = Concatenate(name='gen_dec_deconv_{}_conc'.format(index+1))([X ,en_X[-(index+2)]])
    
    X = Activation("relu",name='gen_dec_conv_last_act')(X)
    X = Conv2DTranspose(nb_channels, kernel_size,strides=strides, name="gen_dec_conv_last", padding="same")(X)
    X = Activation("tanh",name='gen_dec_conv_final_act')(X)

    gen_unet_model = Model(inputs=[In], outputs=[X])

    return gen_unet_model
#--------------------------------------------------------------------------------------
def inception_bn(X, nb_filters=16, kernel_sizes=[(1,1), (3,3), (5,5)]) :
    CXs = []
    for kernel_size in kernel_sizes :
        CX = Conv2D( nb_filters,kernel_size, activation='linear', padding='same')(X)
        CXs.append(CX)
    if (len(CXs)>1):
        X = Concatenate( axis=-1)(CXs)
    else :
        X = CXs[0]
    X= BatchNormalization()(X)
    X= Activation('relu')(X)
    return X

def lambda_fcn(X):
    _,nb_rows,nb_cols,_=K.int_shape(X)
    return tf.image.resize(X,tf.constant([nb_rows*2,nb_cols*2],dtype = tf.int32),align_corners=True)
def lambda_out(in_shape):
    return tuple([in_shape[0],in_shape[1]*2,in_shape[2]*2,in_shape[3]])
#-----------------------------------------------------------------------------------------------
def man_net(img_dim=256,nb_channels=3):
    nb_filters=[64,64,128,128,256,256,256,512,512,512]
    pool_idx=[1,3,6,9]
    nb_f=[i for i in range(8,0,-2)]
    
    img_shape=(img_dim,img_dim,nb_channels)
    IN=Input(shape=img_shape)
    for i in range(len(nb_filters)):
        if i==0:
            X_prev=IN
        X = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same')(X_prev)
        if i in pool_idx:
            X= MaxPooling2D((2, 2), strides=(2, 2))(X)
        X_prev=X
    
    for i in range(len(nb_f)):
        X=inception_bn(X,nb_f[i])
        X=Lambda(lambda_fcn,output_shape=lambda_out)(X)
    
    X= inception_bn(X,nb_filters=2,kernel_sizes=[(5,5),(7,7),(11,11)])
    X = Conv2D(nb_channels, (3,3), activation='sigmoid', padding='same')(X)
    model = Model(inputs=IN, outputs=X)
    return model
#-----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    model=unet()
    model.summary()
    plot_model(model,to_file=os.path.join(os.getcwd(),'info','u-Net.png') ,show_shapes=True)
    
    model=man_net()
    model.summary()
    plot_model(model,to_file=os.path.join(os.getcwd(),'info','man_net.png') ,show_shapes=True)