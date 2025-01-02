#!/bin/bash

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/usr/local/cuda-11.3

python fine_tune_LoRA.py \
  --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml \
  --num-gpus 1