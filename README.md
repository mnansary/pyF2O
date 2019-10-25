# pyF2O
Forged Image To Original Image Generation

    Version: 2.2.0    
    Author : Md. Nazmuddoha Ansary    
                  
![](/INFO/src_img/python.ico?raw=true )
![](/INFO/src_img/tensorflow.ico?raw=true)
![](/INFO/src_img/keras.ico?raw=true)
![](/INFO/src_img/col.ico?raw=true)

# Version and Requirements
* numpy==1.16.4  
* tensorflow==1.14.0        
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
            "MICC-F2000"        : "/home/ansary/RESEARCH/F2O/UNZIPPED/MICC-F2000S/",
            "MICC-F220"         : "/home/ansary/RESEARCH/F2O/UNZIPPED/MICC-F220S/",
            "OUTPUT_DIR"        : "/home/ansary/RESEARCH/F2O/"
        }

**clear_mem.sh (Ubuntu/Linux)**
The complete preprocessing may take huge time and also cause to crash the system due to high memory useage. A way around is built for **Ubuntu** users is to run **sudo ./clear_mem.sh** in parallel with **main.py**

**Results**
* If execution is successful a folder called **DataSet** should be created with the following folder tree:

            DataSet  
            ├── test
            │   ├── image
            │   └── target
            ├── tfrecord
            │   ├── eval
            │   ├── test
            │   └── train
            └── train
                ├── image
                └── target


* The Total Number of data: **16128** (Train=**12928** and Eval=**3200**) + **2640** (Test) 

**ENVIRONMENT**  

    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
    Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
    Gnome       : 3.28.2  

#  GCS
![](/INFO/src_img/bucket.ico?raw=true) TPU training with tfrecord is not implemented for **Tensorflow  1.14.0** as of  **21-10-2019** and  **Tensorflow 2.0** does not have TPU support yet. Hopefully local implementation will be available soon enough. For using TPU in colab, a **bucket** must be created in **GCS** and connected for :
*   saving model checkpoints 
*   loading data

# TPU(Tensor Processing Unit)
![](/INFO/src_img/tpu.ico?raw=true)*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)  


# pix2pix
![](/INFO/p2p.jpg?raw=true)  

Pix2Pix is based on the original paper: [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)
## Acknowledgement
The implementation used here is completely borrowed (with very very minimal changes) from [@agermanidis's implementation of pix2pix-tpu ](https://github.com/agermanidis/pix2pix-tpu)  
For GPU in colab: [Follow This Link](https://www.tensorflow.org/tutorials/generative/pix2pix)   
[Image Source](https://neurohive.io/en/popular-networks/pix2pix-image-to-image-translation/)  



