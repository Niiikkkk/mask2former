#!/bin/bash
export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/usr/local/cuda-11.3

python evaluation_ood_lora.py --input /home/nberardo/Datasets/RoadAnomaly21/images/*.png \
 --config-file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml

#python evaluation_ood_lora.py --input /home/nberardo/Datasets/RoadAnomaly/images/*.jpg \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference_lora.yaml
#
#python evaluation_ood_lora.py --input /home/nberardo/Datasets/RoadObsticle21/images/*.webp \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference_lora.yaml
#
#python evaluation_ood_lora.py --input /home/nberardo/Datasets/FS_LostFound_full/images/*.png \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference_lora.yaml
#
#python evaluation_ood_lora.py --input /home/nberardo/Datasets/fs_static/images/*.jpg \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference_lora.yaml