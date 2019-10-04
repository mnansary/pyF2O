# pyF2O
Forged Image To Original Image Generation

    Version: 1.0.0    
    Author : Md. Nazmuddoha Ansary    
                  
![](/F2O/info/src_img/python.ico?raw=true )
![](/F2O/info/src_img/tensorflow.ico?raw=true)
![](/F2O/info/src_img/keras.ico?raw=true)
![](/F2O/info/src_img/col.ico?raw=true)

# Version and Requirements
* numpy==1.16.4  
* tensorflow==1.13.1        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

#  DataSet
1. Download [Data Sets: MICC-F2000 and MICC-F220](http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/) dataset    
2. Unzip **MICC-F2000.zip** *FOR TRAINING* and **MICC-F220** *FOR TESTING*   
        **The MICC-F2000 dataset contains a file named: nikon7_scale.jpg. It has to be renamed as nikon_7_scale.jpg.**         


#  Preprocessing
**config.json**
 Change The following Values in ***config.json*** 
> data_dir      = Path to the specific Data Folder
> save_dir      = Path to save the processed data
> rename_flag   = If you renamed the file manually from step 2 *(Better way to avoid sys erros)* set this flag to **0** else set **1**  

    "train":   
    {  
        "data_dir"     : "/home/ansary/RESEARCH/F2O/MICC-F2000/", 
        "save_dir"     : "/home/ansary/RESEARCH/F2O/",
        "rename_flag"  : 0,
    
    },  
    "test":  
    {  
        "data_dir"     : "/home/ansary/RESEARCH/F2O/MICC-F220/",
        "save_dir"     : "/home/ansary/RESEARCH/F2O/",
    }        

**clear_mem.sh (Ubuntu/Linux)**
The complete preprocessing may take huge time and also cause to crash the system due to high memory useage. A way around is built for **Ubuntu** users is to run **sudo ./clear_mem.sh** in parallel with **main.py**

**Results**
* If execution is successful a folder called **DataSet** should be created with the following folder tree:

            F2O_DataSet  
            ├── Test
            ├── tfrecords
            │   ├── Eval
            │   └── Train
            └── Train

* The Total Number of data: **16128** (Train=**12928** and Eval=**3200**) + **2640** (Test) 

**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  





