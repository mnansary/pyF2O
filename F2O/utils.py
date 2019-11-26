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
from progressbar import ProgressBar
#--------------------------------------------------------------------------------------------------------------------------------------------------
def LOG_INFO(log_text,p_color='green',rep=True):
    if rep:
        print(colored('#    LOG:','blue')+colored(log_text,p_color))
    else:
        print(colored('#    LOG:','blue')+colored(log_text,p_color),end='\r')


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

def create_dir(base_dir,ext_name):
    new_dir=os.path.join(base_dir,ext_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir
#--------------------------------------------------------------------------------------------------------------------------------------------------
class DataSet(object):
    '''
    This Class is used to preprocess The dataset for Training and Testing
    One single Image is augmented with rotation of (STATICS.rot_angle_start,STATICS.rot_angle_end) increasing by STATICS.rot_angle_step 
    and each rotated image is flipped horizontally,vertically and combinedly to produce 4*N*rot_angles images per one input
    '''
    def __init__(self,data_dir,mode,save_dir,STATICS):
        self.data_dir    = data_dir
        self.mode        = mode
        self.save_dir    = save_dir
        self.STATICS     = STATICS
        # Create DataSet Folder
        self.base_save_dir=create_dir(self.save_dir,'DataSet')
        # mode dir
        self.mode_dir=create_dir(self.base_save_dir,self.mode)
        # image dir
        self.image_dir=create_dir(self.mode_dir,'image')
        # target dir
        self.target_dir=create_dir(self.mode_dir,'target')
        # rename problematic files
        file_to_rename,proper_name=str(self.STATICS.rename_data).split(',')
        if os.path.isfile(os.path.join(self.data_dir,file_to_rename)):
            try:
                os.rename(os.path.join(self.data_dir,file_to_rename),os.path.join(self.data_dir,proper_name))
            except Exception as e:
                print(colored('!!! An exception occurred while renaming {}'.format(file_to_rename),'cyan'))
                print(colored(e,'green'))
        else:
            pass 
              
        # list image paths and idens
        self.prob_idens=str(self.STATICS.prob_idens).split(',')
        self.IMG_Paths=[]
        self.IMG_Idens=[]
        for file_name in glob(os.path.join(self.data_dir,'*{}*.*'.format( self.STATICS.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.STATICS.tamper_iden)]
            if base_name not in self.prob_idens: 
                self.IMG_Paths.append(file_name)
                self.IMG_Idens.append(base_name)
        self.IMG_Idens=list(set(self.IMG_Idens))
        # list ground truth paths and idens
        self.GT_Paths=[]
        self.GT_Idens=[]
        for file_name in glob(os.path.join(self.data_dir,'*{}*.*'.format( self.STATICS.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find( self.STATICS.orig_iden)]
            if base_name in self.IMG_Idens:
                self.GT_Paths.append(file_name)
                self.GT_Idens.append(base_name)

    def create(self):
        _pbar=ProgressBar()
        rotation_angles=[angle for angle in range(self.STATICS.rot_angle_start,self.STATICS.rot_angle_end,self.STATICS.rot_angle_step)]
        for img_path in _pbar(self.IMG_Paths):
            #Get IMG and GT paths
            base_path,_=os.path.splitext(img_path)
            base_name=os.path.basename(base_path)
            iden_name=base_name[:base_name.find( self.STATICS.tamper_iden)]
            gt_path=self.GT_Paths[self.GT_Idens.index(iden_name)]
            # Load IMAGE  
            IMG=imgop.open(img_path)
            # Load GROUNDTRUTH
            GT=imgop.open(gt_path)
            # Crop Data
            IMG,GT=self.__cropData(IMG,GT) 
            # Create Rotations
            for rot_angle in rotation_angles:
                rot_img=IMG.rotate(rot_angle)
                rot_gt=GT.rotate(rot_angle)
                self.__saveTransposedData(rot_img,rot_gt,base_name,rot_angle)            
    
    def __getBbox(self,data):
        rows = np.any(data, axis=1)
        cols = np.any(data, axis=0)
        rmin,rmax = np.where(rows)[0][[0, -1]]
        cmin,cmax = np.where(cols)[0][[0, -1]]
        return rmin,rmax,cmin,cmax
    
    def __pad(self,rmin,rmax,cmin,cmax,diff):
        _pad=16
        cmin -=_pad
        rmin -=_pad
        cmax +=_pad
        rmax +=_pad
        # margin correction
        if cmin < 0:
            cmin = 0
        if rmin < 0:
            rmin = 0
        if cmax > diff.shape[1]:
            cmax=diff.shape[1]
        if rmax > diff.shape[0]:
            rmax=diff.shape[0]
        return rmin,rmax,cmin,cmax
    
    def __cropData(self,IMG,GT):
        # manipulation
        i_o=np.array(IMG)
        g_o=np.array(GT)
        diff= i_o - g_o
        # diff 
        rmin,rmax,cmin,cmax = self.__getBbox(diff)
        rmin,rmax,cmin,cmax = self.__pad(rmin,rmax,cmin,cmax,diff) 
        # cropped Data
        CroppedGT =   np.array(GT.crop((cmin,rmin,cmax,rmax)))
        IMG       =   IMG.crop((cmin,rmin,cmax,rmax))
        # base
        Base    =   imgop.fromarray(diff)
        Base    =   np.array(Base.crop((cmin,rmin,cmax,rmax)))
        # single channel Target
        D1=np.sum(CroppedGT,axis=-1)/(self.STATICS.shade_factor/4)
        D1=D1.astype(np.uint8)
        # array
        D3 = np.zeros(CroppedGT.shape,'uint8')
        # 3 channel Target
        D3[:,:, 0] = D1
        D3[:,:, 1] = D1
        D3[:,:, 2] = D1
        # set cropped data
        rmin,rmax,cmin,cmax=self.__getBbox(Base)
        D3[rmin:rmax+1, cmin:cmax+1]=CroppedGT[rmin:rmax+1, cmin:cmax+1]
        # IMAGE OBJ
        GT=imgop.fromarray(D3)
        # resize
        GT=GT.resize((self.STATICS.image_dim,self.STATICS.image_dim))
        IMG=IMG.resize((self.STATICS.image_dim,self.STATICS.image_dim))
        return IMG,GT 

    def __saveTransposedData(self,rot_img,rot_gt,base_name,rot_angle):
        for fid in range(self.STATICS.fid_num):
            rand_id=random.randint(0,10E4)
            x,y=self.__getFlipDataById(rot_img,rot_gt,fid)
            file_name='{}_{}_fid-{}_angle-{}'.format(rand_id,base_name,fid,rot_angle)
            self.__saveData(x,'image',file_name)
            self.__saveData(y,'target',file_name)
                
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
        return x,y
    
    def __saveData(self,data,identifier,file_name):
        if identifier   =='image':
            save_dir    =   self.image_dir
            #p_color     =   'green'
        elif identifier =='target':
            save_dir    =   self.target_dir
            #p_color     =   'blue'
        else:
            raise ValueError('Identifier not Correct(image/target)')
        file_path=os.path.join(save_dir,file_name+'.png')
        imageio.imsave(file_path,data)
        #LOG_INFO('SAVED {} MODE :{} TYPE:{}'.format(file_name+'.png',self.mode,identifier),p_color=p_color,rep=True) 
    
#--------------------------------------------------------------------------------------------------------------------------------------------------

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,DS,mode,r_num):
    '''
    Creates tfrecords from Provided Image Paths
    Arguments:
    image_paths = List of Image Paths with Fixed Size (NOT THE WHOLE Dataset)
    DS          = DataSet Object  (Only attributes are needed)
    mode        = Mode of data to be created
    r_num       = number of record
    '''
    # Create Saving Directory based on mode
    save_dir=create_dir(DS.base_save_dir,'tfrecord')
    save_dir=create_dir(save_dir,mode)
    #LOG_INFO('TFRECORD:DIR: {}'.format(save_dir),p_color='yellow')
    # Tfrecord Info
    tfrecord_name='{}_{}.tfrecord'.format(mode,r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in image_paths:
            #LOG_INFO('Converting: {} '.format(image_path))
            target_path=str(image_path).replace('image','target')
            #LOG_INFO('Converting: {} '.format(target_path),p_color='cyan')
            with(open(image_path,'rb')) as fid:
                image_png_bytes=fid.read()
            with(open(target_path,'rb')) as fid:
                target_png_bytes=fid.read()
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes)
            }
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)   
    #LOG_INFO('Finished Writing {}'.format(tfrecord_name),p_color='red')

#--------------------------------------------------------------------------------------------------------------------------------------------------
def data_input_fn(FLAGS): 
    '''
    This Function generates data from provided FLAGS
    FLAGS must include:
        TFRECORDS_PATH  = Path to tfrecords
        MODE            = 'train/test'
        IMAGE_DIM       = Dimension of Image
        NB_CHANNELS     = Depth of Image
        BATCH_SIZE      = batch size for traning
        SHUFFLE_BUFFER  = Buffer Size > Batch Size
        DEBUG           = check data and model train for keras simple model
    '''
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'target' : tf.io.FixedLenFeature([],tf.string)
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=FLAGS.NB_CHANNELS)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        target_raw=parsed_example['target']
        target=tf.image.decode_png(target_raw,channels=FLAGS.NB_CHANNELS)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,FLAGS.NB_CHANNELS))
        
        return image,target

    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,FLAGS.MODE,'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    if FLAGS.MODE=='train':
        dataset = dataset.shuffle(FLAGS.SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    if FLAGS.DEBUG:
        dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.BATCH_SIZE,drop_remainder=True)
    return dataset
