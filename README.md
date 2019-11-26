# pyF2O
Forged Image To Original Image Generation

    Version: 3.0.0   
    Author : Md. Nazmuddoha Ansary
             Shakir Hossain  
             Mohammad Bin Monjil  
             Habibur Rahman
             MD.Aminul Islam
             Shahriar Prince  
                  
![](/INFO/src_img/python.ico?raw=true )
![](/INFO/src_img/tensorflow.ico?raw=true)
![](/INFO/src_img/keras.ico?raw=true)
![](/INFO/src_img/col.ico?raw=true)

# Version and Requirements
* numpy==1.17.4  
* tensorflow==2.0.0        
* Python == 3.6.8
> Create a Virtualenv and *pip3 install -r requirements.txt*

#  DataSet
1. Download [Data Sets: MICC-F2000 and MICC-F220](http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/) dataset    
2. Unzip **MICC-F2000.zip** *FOR TRAINING* and **MICC-F220** *FOR TESTING*   
        **The MICC-F2000 dataset contains a file named: nikon7_scale.jpg. It has to be renamed as nikon_7_scale.jpg.**         

#  Preprocessing
**config.json**
 Change The following Values in ***config.json*** 

        "ARGS":
        {
            "MICC-F2000"        : "/home/ansary/RESEARCH/F2O/UNZIPPED/MICC-F2000/",
            "MICC-F220"         : "/home/ansary/RESEARCH/F2O/UNZIPPED/MICC-F220/",
            "OUTPUT_DIR"        : "/home/ansary/RESEARCH/F2O/"
        }

**clear_mem.sh (Ubuntu/Linux)**
The complete preprocessing may take huge time and also cause to crash the system due to high memory useage. A way around is built for **Ubuntu** users is to run **sudo ./clear_mem.sh** in parallel with **main.py**


            usage: main.py [-h] exec_flag

            Preprocessing Script:Forged Image To Original Image Reconstruction

            positional arguments:
            exec_flag   
                                                    Execution Flag for creating files 
                                                    Available Flags: png,tfrecords,comb
                                                    png       = create images
                                                    tfrecords = create tfrecords
                                                    comb      = combined execution
                                                    PLEASE NOTE:
                                                    For Separate Run the following order must be maintained:
                                                    1) png
                                                    2) tfrecords
                                                    
                                                    

            optional arguments:
            -h, --help  show this help message and exit




**Results**
* If execution is successful a folder called **DataSet** should be created with the following folder tree:

            DataSet  
            ├── test
            │   ├── image
            │   └── target
            ├── tfrecord
            │   ├── test
            │   └── train
            └── train
                ├── image
                └── target



**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  

#  GCS	
![](/INFO/src_img/bucket.ico?raw=true) Training with tfrecord is not implemented for local implementation.	
For using colab, a **bucket** must be created in **GCS** and connected for:
* tfrecords
* checkpoints 	

# Networks

* **Generator** structre

![](/INFO/gen.png?raw=true)  

* **Discriminator** structre

![](/INFO/dis.png?raw=true)  

# pix2pix

![](/INFO/p2p.jpg?raw=true)  

[Image Source](https://neurohive.io/en/popular-networks/pix2pix-image-to-image-translation/)    

Original paper: [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)  

Implementation based on [official tensorflow tutorial](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)  

* run **pix2pix_gpu.ipynb** in *colab*





