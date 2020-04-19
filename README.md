# T4M5
**Module 5 of Master of Computer Vision -  Team 4**

#### Authors - Team04
- _E. Mainou, eduardmgobierno@gmail.com - [EddieMG](https://github.com/EddieMG)_
- _H. Valenciano, hermes.valenciano1996@gmail.com - [hermesvf](https://github.com/hermesvf)_
- _P. Albacar, pol.albacar@hotmail.com - [polalbacar](https://github.com/polalbacar)_
- _S. Morawski, malpunek@gmail.com - [malpunek](https://github.com/malpunek)_


# Week 1 - Introduction to Pytorch

The main task in to implement the final model from M3 (Image Classification) in Pytorch.

The slides reporting the results obtained can be seen in [these Google Slides](https://docs.google.com/presentation/d/1n3wsHfT0nL-1LW-X02lgyh8OLpLObmZt5nEjD4H5RaE/edit#slide=id.p).

The code to run this task is the **main.py** that can be found at **w1** folder.


# Week 2 - Introduction to Object Detection
Main tasks:

(a) Getting familiar with Detectron2 framework

(b) Use object detection models in inference: Faster RCNN

(c) Use object detection models in inference: RetinaNet

(d) Train a Faster R-CNN on KITTI dataset

The slides reporting the results obteined can be seen in [these Google Slides](https://docs.google.com/presentation/d/14V1yCVCiOaDklb_7u7BKgAKRDSIhGrjze1V1CBX6Qm0/edit#slide=id.g7102907464_5_34).

The code to run these tasks are the **task_a.py task_b.py task_c.py task_d.py** an be found at **w2** folder.


# Week 3 - Multiple Object Tracking and Segmentation

Main tasks:

(a) Get familiar with MOTSChallenge and KITTI-MOTS challenges

(b) Use object pre-trained object detection models in inference on KITTI-MOTS

(c) Evaluate pre-trained object detection models on KITTI-MOTS 

(d) Train detection models on KITTI-MOTS and MOTSChallenge training sets

(e) Evaluate best trained model on KITTI-MOTS test set

The slides reporting the results obteined can be seen in [these Google Slides](https://docs.google.com/presentation/d/1rppl8bJZF5lnt4Qxvoe_KrF_eDC2S-eNhT6g58L_NlE/edit#slide=id.g7168e8d968_29_0).

The code to run these tasks are the **task_b_faster.py task_b_retina.py task_c_faster.py task_c_retina.py task_d.py** an be found at **w3** folder.


# Week 4 - Introduction to Object Segmentation

Main tasks:

(a) Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set. Trying different configurations:
  - [x] Number of layers
  - [x] Backbone configuration
  - [x] Use of Feature Pyramid Network
  - [x] Use of training data (COCO vs COCO+Cityscapes)

(b) Train Mask-RCNN model on KITTI-MOTS training set and evaluate on KITTI-MOTS validation set
  - [x] COCO + KITTI-MOTS
  - [x] COCO + Cityscapes + KITTI-MOTS

The slides reporting the results obteined can be seen in [these Google Slides](https://docs.google.com/presentation/d/1Wxv_nS51v2C9CKlNpzeHORPC9lifEhkCmpZSD9jJOXA/edit#slide=id.g718556d907_1_0).

The code to run all tasks together is the **main.py** that can be found at **w4** folder.


# Week 5 - Transfer Learning for Object Detection and Segmentation

Main tasks:

(a) Apply pre-trained and finetuned Mask-RCNN models to MOTSChallenge train set
  - [x] COCO
  - [x] COCO+Cityscapes
  - [x] COCO+KITTI-MOTS
  - [x] COCO+Cityscapes+KITTI-MOTS

(b) Apply pre-trained and finetuned Mask-RCNN models to KITTI-MOTS val set
  - [x] COCO + MOTSChallenge
  - [x] COCO + Cityscapes + MOTSChallenge
  - [x] COCO + MOTSChallenge + KITTI-MOTS
  - [x] COCO + Cityscapes + MOTSChallenge + KITTI-MOTS

(c) Explore and analyze the impact of different hyperparameters. We have managed to try these ones:
  - [x] IOU_THRESHOLD
  - [x] LR_SCHEDULER
  - [x] ANCHOR_GENERATOR.SIZES

The slides reporting the results obteined can be seen in [these Google Slides](https://docs.google.com/presentation/d/1GoxeIPR7aRU02mNyxeSnqRkAa7uV55FtJaIlOdWdFMM/edit#slide=id.g72363b1db2_18_3).

The code to run tasks a and b together is the **task_a_b.py** that can be found at **w5** folder. Hyperparameters for task c can be changed on the task_b_MOTS_and_KITTI_training function placed at task_a_b.py file.


# Week 6 - Data Augmentation, Semantic Segmentation and Video Object Segmentation

Main tasks:

(a) Add data augmentation techniques to Detectron2 framework

(b) Train your model on a synthetic dataset and finetune it on a real dataset

(c) Train a semantic segmentation model (DeepLabv3)

The slides reporting the results obteined can be seen in [these Google Slides](https://docs.google.com/presentation/d/1ydBIwr2Vx4eIkHH6BRrn0nSDtqjCqCaG16S4zq_4Se8/edit#slide=id.g7350972f6d_0_0).

The code to run these tasks are the **task_a.py task_b.py task_c.py** an be found at **w6** folder.


# Paper
The paper will be updated weekly according to the work done in the project.
[M5 Project: Weekly paper](https://www.overleaf.com/read/zwjphfsmqyjt)

The final paper (compated in 4 pages): [M5 Project: Final paper](https://www.overleaf.com/read/zwjphfsmqyjt)


