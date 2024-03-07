# Diversify-and-Aggregate: Augmenting Replay with Generative Modeling Make Stronger Incremental Segmentation Models

This is an official implementation of the paper "Diversify-and-Aggregate: Augmenting Replay with Generative Modeling Make Stronger Incremental Segmentation Models".

<!-- For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/DKD/)] and our paper [[arXiv](http://arxiv.org/abs/2210.05941) / [OpenReview](https://openreview.net/forum?id=0SgKq4ZC9r)]. -->

## Pre-requisites
This repository has been tested with the following libraries:
* Python (3.9)
* Pytorch (2.2.0)

## Getting Started

### Datasets
#### PASCAL VOC 2012
We use augmented 10,582 training samples and 1,449 validation samples for PASCAL VOC 2012. You can download the original dataset in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit). To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). The structure of data path should be organized as follows:
```bash
└── /dataset/VOC2012
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation
    │       ├── train_aug.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationClassAug
```

#### ADE20K
We use 20,210 training samples and 2,000 validation samples for ADE20K. You can download the dataset in [here](http://sceneparsing.csail.mit.edu/). The structure of data path should be organized as follows:
```bash
└── /dataset/ADEChallengeData2016
    ├── annotations
    ├── images
    ├── objectInfo150.txt
    └── sceneCategories.txt
```

### Training
#### PASCAL VOC 2012
```Shell
# An example srcipt for 15-5 overlapped setting of PASCAL VOC

GPU=0,1
BS=16  # Total 32
SAVEDIR='saved_voc_pos2'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-5' # or ['15-1', '19-1', '10-1', '5-3']
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=100 

NAME='DA'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight_new ${INIT_POSWEIGHT}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE} --pos_weight_new 1 --pos_weight_old 1 --pkd 5 --mbce_new_extra 1 --mbce_old_extra 1 --use_Replace
```

#### ADE20K
```Shell
# An example srcipt for 50-50 overlapped setting of ADE20K

GPU=0,1
BS=12  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='50-50' # or ['100-10', '100-50']
EPOCH=100
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=300

NAME='DA'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE} --pos_weight_new 1 --pos_weight_old 1 --pkd 1 --mbce_new_extra 1 --mbce_old_extra 1 --use_Replace
```

### Testing
#### PASCAL VOC 2012
```Shell
python eval_voc.py -d 0 -r path/to/weight.pth
```

We provide pretrained weights, augmented images and adapter checkpoint (lora, text token) and configuration files. 

 - [x] configuration files.
 - [x] pretrained weights, 
 - [x] augmented images and adapter checkpoint (lora, text token)
 - [ ] code for fine-tuning MR-LoRA (coming soon)

## Acknowledgements
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
* This code is based on DKD ([2022-NeurIPS](https://github.com/cvlab-yonsei/DKD) Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation).
