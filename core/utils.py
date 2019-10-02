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

import h5py
import tensorflow as tf 
import time
#--------------------------------------------------------------COMMON------------------------------------------------------------------------------------
def LOG_INFO(log_text,p_color='green'):
    print(colored('#    LOG:','blue')+colored(log_text,p_color))

def saveh5(path,data):
    hf = h5py.File(path,'w')
    hf.create_dataset('data',data=data)
    hf.close()

def readh5(d_path):
    data=h5py.File(d_path, 'r')
    data = np.array(data['data'])
    return data

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
    One single Image is augmented with rotation of 0 and 45 degree 
    and each rotated image is flipped horizontally,vertically and combinedly to produce 8 images per one input
    
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

        self.base_save_dir=os.path.join(args.save_dir,'F2O_DataSet')
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
            raise ValueError('Wrong Dataset!!!')
        

        self.save_dir=os.path.join(self.base_save_dir,self.mode)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

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
        
        return np.concatenate((x,y),axis=1)

    def __saveData(self,data,identifier):
        file_name=os.path.join(self.save_dir,identifier+'.png')
        LOG_INFO('Saving {} MODE: {} DATASET: {}'.format(identifier+'.png',self.mode,self.dataSet)) 
        imageio.imsave(file_name,data)    

    def __saveTransposedData(self,rot_img,rot_gt,base_name,rot_angle):
        for fid in range(4):
            rand_id=random.randint(0,20000)
            data=self.__getFlipDataById(rot_img,rot_gt,fid)
            self.__saveData(data,'{}_{}_fid-{}_angle-{}'.format(rand_id,base_name,fid,rot_angle))

    def __genDataSet(self):
        rotation_angles=[0,45]
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


def to_tfrecord(image_paths,obj,r_num):
    '''
    Creates tfrecords from Provided Image Paths
    Arguments:
    image_paths = List of Image Paths with Fixed Size (NOT THE WHOLE Dataset)
    obj         = DataSet Object  (Only attributes are needed)
    r_num       = number of record
    '''
    # Create Saving Directory based on mode
    save_dir=os.path.join(obj.base_save_dir,'tfrecords')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_dir=os.path.join(save_dir,obj.mode)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    LOG_INFO('TFRECORD:DIR: {}'.format(save_dir),p_color='yellow')
    
    # Tfrecord Info
    tfrecord_name='{}_{}.tfrecords'.format(obj.mode,r_num)
    tfrecord_path=os.path.join(save_dir,tfrecord_name) 
    
    for image_path in image_paths:
        LOG_INFO('Converting: {} to protocol buffer'.format(image_path))
    
        img = imgop.open(image_path)
        arr = np.array(img.resize((2*obj.image_dim,obj.image_dim)))
        IMG = arr[:, :obj.image_dim, :]
        GT = arr[:, obj.image_dim:, :]
    
        feature ={ 'target': _bytes_feature(GT.tostring()),
                    'image': _bytes_feature(IMG.tostring()) 
        }
    
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            example_protocol_buffer = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_protocol_buffer.SerializeToString())
        
    LOG_INFO('Finished Writing {}'.format(tfrecord_name),p_color='red')
#--------------------------------------------------------------------------------------------------------------------------------------------------
def get_batched_dataset(FLAGS):
    '''
    This Function generates data from provided FLAGS
    FLAGS must include:
    TFRECORDS_DIR   = Directory of tfrecords
    BATCH_SIZE      = Batch Size of the data
    NUM_EPOCHS      = Total Number of epochs
    IMAGE_DIM       = Dimension of Image
    SHUFFLE_BUFFER  = Size for shuffle buffer > batch size
    '''
    def _parser(example_protocol_buffer):
        feature ={  'target'  : tf.io.FixedLenFeature([],tf.string) ,
                    'image'   : tf.io.FixedLenFeature([],tf.string)
        }
        parsed_feature=tf.parse_single_example(example_protocol_buffer,feature)
        
        x =   tf.image.decode_png(parsed_feature['image'],channels=3)
        x =   tf.cast(x,tf.float32)/255.0
        x =   tf.reshape(x,[FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,3])
 
        y =   tf.image.decode_png(parsed_feature['target'],channels=3)
        y =   tf.cast(y,tf.float32)/255.0
        y =   tf.reshape(y,[FLAGS.IMAGE_DIM,FLAGS.IMAGE_DIM,3])

        return x,y 
    
    file_paths=glob(os.path.join(FLAGS.TFRECORDS_DIR,'*.tfrecords'))

    dataset =   tf.data.TFRecordDataset(file_paths)
    dataset =   dataset.map(_parser)
    dataset =   dataset.repeat()
    dataset =   dataset.shuffle(FLAGS.SHUFFLE_BUFFER)
    dataset =   dataset.batch(FLAGS.BATCH_SIZE)
    
    iterator = dataset.make_one_shot_iterator()
    image, target = iterator.get_next()

    return image,target

#--------------------------------------------------------------------------------------------------------------------------------------------------
def H5_data(image_path_list,obj,r_num):
    start_time=time.time()
    X=[]
    Y=[]
    for image_path in image_path_list:
    
        LOG_INFO('IMG: {} '.format(image_path))
    
        img_obj=imgop.open(image_path)
        
        arr = np.array(img_obj.resize((2*obj.image_dim,obj.image_dim)))
        
        x = arr[:, :obj.image_dim, :]
        y = arr[:, obj.image_dim:, :]

        X.append(np.expand_dims(x,axis=0))
        Y.append(np.expand_dims(y,axis=0))
    
    LOG_INFO('Appending Time Taken: {} s'.format(time.time()-start_time),'yellow')
    start_time=time.time()
    
    X=np.vstack(X)
    
    LOG_INFO('X stacking Time Taken: {} s'.format(time.time()-start_time),'yellow')
    start_time=time.time()
    
    Y=np.vstack(Y)
    
    LOG_INFO('Y stacking Time Taken: {} s'.format(time.time()-start_time),'yellow')
    start_time=time.time()
    
    h5_dir=os.path.join(obj.base_save_dir,'H5Data')
    if not os.path.exists(h5_dir):
        os.mkdir(h5_dir)

    X_p=os.path.join(h5_dir,'X_{}_{}.h5'.format(obj.mode,r_num))
    
    Y_p=os.path.join(h5_dir,'Y_{}_{}.h5'.format(obj.mode,r_num))
    
    saveh5(X_p,X)
    
    LOG_INFO('X H5 Saivng Time Taken: {} s'.format(time.time()-start_time),'yellow')
    start_time=time.time()
    
    saveh5(Y_p,Y)
    
    LOG_INFO('Y H5 Saving Time Taken: {} s'.format(time.time()-start_time),'yellow')
    
    return X_p,Y_p
    