"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

import os
import numpy as np 
from glob import glob
import random
import matplotlib.pyplot as plt

from PIL import Image as imgop
import imageio

import tensorflow as tf 
import time

from functools import partial
#--------------------------------------------------------------COMMON------------------------------------------------------------------------------------
def LOG_INFO(log_text,p_color='green'):
    print(colored('#    LOG:','blue')+colored(log_text,p_color))

def plot_data(img,gt,pred,net,save_flag=None,show_imdt=False) :
    plt.figure(net)
    plt.subplot(131)
    plt.imshow(img)
    plt.title(' image')
    plt.subplot(132)
    plt.title('ground truth')
    plt.imshow(gt)
    plt.subplot(133)
    plt.imshow(pred)
    plt.title('prediction')
    
    if save_flag:
        plt.savefig(save_flag)
    if show_imdt:
        plt.show()
    
    plt.clf()
    plt.close()

#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    '''
    This Class is used to preprocess The dataset for Training and Testing
    One single Image is augmented with rotation of (0,25) increasing by 5 degree 
    and each rotated image is flipped horizontally,vertically and combinedly to produce 24 images per one input
    
    args must include:
    data_dir    = Directory of The Unzipped MICC-XXXX folder
    save_dir    = Directory for Saving the preprocessed Data
    image_dim   = Dimension of Image Resize
    dataset_name= Name of the DataSet (i.e- MICC-F2000 / MICC-F220)
    rename_flag = (True/False) If you have not manually renamed problematic Image ,set this Flag To False
                  (Visit- https://github.com/mnansary/pyF2O/blob/master/README.md for further clarification )
    tamper_iden = Tampered Image Identifier
    orig_iden   = Original Image Identifier 
    
    '''
    def __init__(self,args):
        self.data_dir=args.data_dir

        self.base_save_dir=os.path.join(args.save_dir,'DataSet')
        if not os.path.exists(self.base_save_dir):
            os.mkdir(self.base_save_dir)

        self.image_dim=args.image_dim
        self.tamper_iden=args.tamper_iden
        self.orig_iden=args.orig_iden

        self.dataSet=args.dataset_name
        if self.dataSet=='MICC-F2000':
            self.mode='Train'
            if args.rename_flag==1:
                self.__renameProblematicFileMICC_F2000()

        elif self.dataSet=='MICC-F220':
            self.mode='Test'
        else:
            self.mode='Eval'
        
        self.save_dir=os.path.join(self.base_save_dir,self.mode)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        

        self.prb_idens=['P1000231','DSCN47']
        self.batch_size=args.batch_size

    def __renameProblematicFileMICC_F2000(self):
        file_to_rename='nikon7_scale.jpg'
        proper_name='nikon_7_scale.jpg'
        try:
            os.rename(os.path.join(self.data_dir,file_to_rename),os.path.join(self.data_dir,proper_name))
        except Exception as e:
            print(colored('!!! An exception occurred while renaming {}'.format(file_to_rename),'cyan'))
            print(colored(e,'green'))
        
    def __listFiles(self):
        
        self.IMG_Paths=[]
        self.IMG_Idens=[]
        for file_name in glob(os.path.join(self.data_dir,'*{}*.*'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            if base_name not in self.prb_idens: 
                self.IMG_Paths.append(file_name)
                self.IMG_Idens.append(base_name)
            
        self.IMG_Idens=list(set(self.IMG_Idens))
        self.GT_Paths=[]
        self.GT_Idens=[]
        
        for file_name in glob(os.path.join(self.data_dir,'*{}*.*'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            if base_name in self.IMG_Idens:
                self.GT_Paths.append(file_name)
                self.GT_Idens.append(base_name)


    def __getFlipDataById(self,img,gt,fid):
        if fid==0:# ORIGINAL
            x=np.array(img)
            y=np.array(gt)
        elif fid==1:# Left Right Flip
            x=np.array(img.transpose(imgop.FLIP_LEFT_RIGHT))
            y=np.array(gt.transpose(imgop.FLIP_LEFT_RIGHT))
        elif fid==2:# Up Down Flip
            x=np.array(img.transpose(imgop.FLIP_TOP_BOTTOM))
            y=np.array(gt.transpose(imgop.FLIP_TOP_BOTTOM))
        else: # Mirror Flip
            x=img.transpose(imgop.FLIP_TOP_BOTTOM)
            x=np.array(x.transpose(imgop.FLIP_LEFT_RIGHT))
            y=gt.transpose(imgop.FLIP_TOP_BOTTOM)
            y=np.array(y.transpose(imgop.FLIP_LEFT_RIGHT))
        
        d=x-y
        d[d!=0]=y[d!=0]
        return x,d

    def __saveData(self,data,identifier):
        file_name=os.path.join(self.save_dir,identifier+'.png')
        LOG_INFO('Saving {} MODE: {} DATASET: {}'.format(identifier+'.png',self.mode,self.dataSet)) 
        imageio.imsave(file_name,data)    

    def __saveTransposedData(self,rot_img,rot_gt,base_name,rot_angle):
        for fid in range(4):
            rand_id=random.randint(0,20000)
            x,y=self.__getFlipDataById(rot_img,rot_gt,fid)
            self.__saveData(x,'{}_{}_fid-{}_angle-{}_image'.format(rand_id,base_name,fid,rot_angle))
            self.__saveData(y,'{}_{}_fid-{}_angle-{}_target'.format(rand_id,base_name,fid,rot_angle))
            

    def __genDataSet(self):
        rotation_angles=[i for i in range(0,30,5)]
        for img_path in self.IMG_Paths:
            #Get IMG and GT paths
            base_path,_=os.path.splitext(img_path)
            base_name=os.path.basename(base_path)
            iden_name=base_name[:base_name.find(self.tamper_iden)]
            gt_path=self.GT_Paths[self.GT_Idens.index(iden_name)]
            # Load IMAGE and GROUNDTRUTH
            IMG=imgop.open(img_path)
            GT=imgop.open(gt_path)
            
            # Resize 
            IMG=IMG.resize((self.image_dim,self.image_dim))
            GT=GT.resize((self.image_dim,self.image_dim))
            
            # Create Rotations
            for rot_angle in rotation_angles:
                rot_img=IMG.rotate(rot_angle)
                rot_gt=GT.rotate(rot_angle)
                self.__saveTransposedData(rot_img,rot_gt,base_name,rot_angle)
                
    def preprocess(self):
        LOG_INFO('Preprocessing MODE: {} from {}'.format(self.mode,self.dataSet),p_color='cyan')
        self.__listFiles()
        self.__genDataSet()
        
#--------------------------------------------------------------------------------------------------------------------------------------------------

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,obj,r_num):
    '''
    Creates tfrecords from Provided Image Paths
    Arguments:
    image_paths = List of Image Paths with Fixed Size (NOT THE WHOLE Dataset)
    obj         = DataSet Object  (Only attributes are needed)
    r_num       = number of record
    '''
    # Create Saving Directory based on mode
    save_dir=os.path.join(obj.base_save_dir,'tfrecord')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_dir=os.path.join(save_dir,obj.mode)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    LOG_INFO('TFRECORD:DIR: {}'.format(save_dir),p_color='yellow')
    
    # Tfrecord Info
    tfrecord_name='{}_{}.tfrecord'.format(obj.mode,r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name) 
    
    for image_path in image_paths:
        if '_image' in image_path:
            LOG_INFO('Converting: {} '.format(image_path))
            target_path=str(image_path).replace('_image','_target')
            LOG_INFO('Converting: {} '.format(target_path),p_color='cyan')
            
            with(open(image_path,'rb')) as fid:
                image_png_bytes=fid.read()

            with(open(target_path,'rb')) as fid:
                target_png_bytes=fid.read()

            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes)
            }
        
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)
            
    LOG_INFO('Finished Writing {}'.format(tfrecord_name),p_color='red')
#--------------------------------------------------------------------------------------------------------------------------------------------------

def data_input_fn(FLAGS,MODE): 
    '''
    This Function generates data from provided FLAGS
    FLAGS must include:
        TFRECORDS_DIR   = Directory of tfrecords
        IMAGE_DIM       = Dimension of Image
        NB_CHANNELS     = Depth of Image
        BATCH_SIZE      = batch size for traning
        SHUFFLE_BUFFER  = Buffer Size > Batch Size
        EPOCHS          = Num of epochs to repeat the dataset
    '''
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'target' : tf.io.FixedLenFeature([],tf.string)
        }
        parsed_example=tf.parse_single_example(example,feature)
        
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=FLAGS.NB_CHANNELS)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        target_raw=parsed_example['target']
        target=tf.image.decode_png(target_raw,channels=FLAGS.NB_CHANNELS)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        return image,target 
    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,MODE,'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(FLAGS.SHUFFLE_BUFFER)
    dataset = dataset.repeat(FLAGS.EPOCHS)
    dataset = dataset.batch(FLAGS.BATCH_SIZE, drop_remainder=True)
    return dataset

#--------------------------------------------------------------------------------------------------------------------------------------------------
