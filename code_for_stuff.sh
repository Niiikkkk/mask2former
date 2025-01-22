#!/bin/bash

#python code_for_stuff.py \
#  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
#  --input /home/nberardo/Datasets/RoadAnomaly21/images/*.png \
#
#python code_for_stuff.py \
#--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
#--input /home/nberardo/Datasets/RoadAnomaly/images/*.jpg \
#
#python code_for_stuff.py \
#--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
#--input /home/nberardo/Datasets/RoadObsticle21/images/*.webp \

python code_for_stuff.py \
--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
--input /home/nberardo/Datasets/FS_LostFound_full/images/*.png \

#python code_for_stuff.py \
#--config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml \
#--input /home/nberardo/Datasets/fs_static/images/*.jpg \

