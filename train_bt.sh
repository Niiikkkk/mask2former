#!/bin/bash

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/usr/local/cuda-11.3

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/simsiam/resnet50.pth /home/nberardo/mask2anomaly_v2/backbone_weights/simsiam_resnet50.pkl

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml \
  --num-gpus 1
