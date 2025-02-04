#!/bin/bash

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/usr/local/cuda-11.3

#python code_for_stuff.py \
#  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml \
#  --num-gpus 1

python code_for_stuff.py \
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
--input /home/nberardo/Datasets/FS_LostFound_full/images/*.png \

