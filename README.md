# pyF2O
Forged Image To Original Image Generation

    Version: 0.0.1    
    Author : Md. Nazmuddoha Ansary    
                  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements
* numpy==1.16.4  
* tensorflow==1.13.1        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

#  DataSet
1. Download [Data Sets: MICC-F2000 and MICC-F220](http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/) dataset    
2. Unzip **MICC-F2000.zip** *FOR TRAINING* and **MICC-F220** *FOR TESTING*   
        **The MICC-F2000 dataset contains a file named: nikon7_scale.jpg. It has to be renamed as nikon_7_scale.jpg.**         
3. Change The following Values in ***config.json*** 
> data_dir      = Path to the specific Data Folder
> save_dir      = Path to save the processed data
> rename_flag   = If you renamed the file manually from step 2 *(Better way to avoid sys erros)* set this flag to **0** else set **1**  

    "train":   
    {  
        "data_dir"     : "/home/ansary/RESEARCH/CopyMove/Data/MICC-F2000/", 
        "save_dir"     : "/home/ansary/RESEARCH/F2O/",
        "rename_flag"  : 0,
    
    },  
    "test":  
    {  
        "data_dir"     : "/home/ansary/RESEARCH/CopyMove/Data/MICC-F220/",
        "save_dir"     : "/home/ansary/RESEARCH/F2O/",
    }        

4. Run **./dataset.py**
* If execution is successful a folder called **F2O_DataSet** should be created with the following folder tree depending on the **PARAMS**:  

        F2O_DataSet  
        ├── H5  
        │   ├── X_Test.h5  
        │   ├── X_Train.h5  
        │   ├── Y_Test.h5  
        │   └── Y_Train.h5  
        ├── Test  
        │   ├── XXXXXX.png  
        │   ....(Test Images)  
        │   └── XXXXXX.png  
        ├── tfrecords  
        │   ├── Test.tfrecords  
        │   └── Train.tfrecords  
        └── Train  
            ├── XXXXXX.png  
            ....(Train Images)      
            └── XXXXXX.png  

* The *.h5* files are created for training with ***tf.keras*** (TPU)
* The *.tfrecords* are created for training with ***Estimators*** (TPU and CPU) *WILL BE DONE IN THE FUTURE*
* The *'.png'* is a concatenated version of **ForgedImage|GroundTruth**
> To understand the charecteristics of the *.png* data please read the Docstrings of **DataSet** class from  **models/utils.py**  

**NOTE:** Please wait patiently as the execution may take quite some time to be completed.For safe operation:
1. Set **"create_dataset"** = **1** and **save_h5** and **save_tfrecord**=0 in ***config.json** . Run **dataset.py** to create the dataset images first.
2. Set **save_h5**=**1** and **"create_dataset"** and **save_tfrecord**=0 in ***config.json** . Run **dataset.py** again to create the h5 files.


**ENVIRONMENT DETAILS FOR dataset.py EXECUTION**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  
