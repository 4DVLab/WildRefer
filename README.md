# WildRefer: 3D Object Localization in Large-scale Dynamic Scenes with Multi-modal Visual Data and Natural Language
### [Project Page](https://4dvlab.github.io/project_page/wildrefer.html)
This repository is for WildRefer dataset and official implement for WildRefer: **WildRefer: 3D Object Localization in Large-scale Dynamic Scenes with Multi-modal Visual Data and Natural Language**.

# Dataset
Our dataset can be download [here](https://huggingface.co/datasets/Albert-2024/wildrefer)

We strongly recommend to use our pre-processed [HuCenLife](https://github.com/4DVLab/HuCenLife) and [STCrowd](https://github.com/4DVLab/STCrowd) that can be downloaded [here](https://drive.google.com/drive/folders/1g5OgIT3svL6TPXcusPPsigsnP_Yi1JT7?usp=drive_link).

# How to use this code
## Data Preparation
Please prepare the dataset as following folder struction:
```
./
└── data/
    ├── liferefer_test.json
    ├── liferefer_train.json
    ├── strefer_test.json        
    └── strefer_train.json    
└── src/      
    ├── LifeRefer.zip
    └── STRefer.zip
```
Unzip our processed data
```bash
cd src
unzip LifeRefer.zip
unzip STRefer.zip
cd ..
```

## Environment Installation
Our environment is based on Python 3.8 and cuda 11.3.
You can install the environment with `conda`.
```bash
conda create -n wildrefer_env python=3.8 -y
conda activate wildrefer_env
conda install conda-forge::cudatoolkit-dev=11.3 -y
pip install torch==1.11.0 torchvision==0.12.0 --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd pointnet2
python setup.py install
cd ..
```

## Test
Our weights can be downloaded [here](https://drive.google.com/drive/folders/1SF0Uh-ZhsYFIe48iPDtNX_HDWGraAWwJ?usp=drive_link).
You can put the weights under the folder `weights/`.
```
./
└── weights/
    ├── liferefer_test.json       
    └── strefer_train.json    
```
### STRefer
```bash
python test.py --dataset strefer --pretrain weights/strefer_weights.pth --max_lang_num 50 --frame_num 2 --batch_size 36 
```

### LifeRefer
```bash
python test.py --dataset liferefer --pretrain weights/liferefer_weights.pth --frame_num 2 --batch_size 32
```

## Train
### STRefer
```bash
python train.py --dataset strefer --max_lang_num 50
```

### LifeRefer
```bash
python train.py --dataset liferefer --max_lang_num 100
```

##  License:
All datasets are published under the [Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. 
