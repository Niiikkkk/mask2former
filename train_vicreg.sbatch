#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2a_vicreg
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2anomaly_v2/train_vicreg.log
#SBATCH --error=/home/nberardo/mask2anomaly_v2/train_vicreg.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/mask2anomaly_v2/backbone_weights/vicreg_resnet50_top.pth /home/nberardo/mask2anomaly_v2/backbone_weights/vicreg_resnet50_top.pkl

#train on the pkl weights
#CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_train.yaml

#fine-tune
#CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_ft.yaml

## Anomaly Inference
#RoadAnomaly21
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/nberardo/Datasets/RoadAnomaly21/images/*.png' --config-file '/home/nberardo/mask2anomaly_v2/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' --output '/home/nberardo/mask2anomaly_v2/output/result_eval/vicreg/'
#RoadObsticle21
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/nberardo/Datasets/RoadObsticle21/images/*.webp' --config-file '/home/nberardo/mask2anomaly_v2/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' --output '/home/nberardo/mask2anomaly_v2/output/result_eval/vicreg/'
#FS L&F val
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/nberardo/Datasets/FS_LostFound_full/images/*.png' --config-file '/home/nberardo/mask2anomaly_v2/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' --output '/home/nberardo/mask2anomaly_v2/output/result_eval/vicreg/'
# FS-Static
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/nberardo/Datasets/fs_static/images/*.jpg' --config-file '/home/nberardo/mask2anomaly_v2/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' --output '/home/nberardo/mask2anomaly_v2/output/result_eval/vicreg/'
#RoadAnomaly
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/home/nberardo/Datasets/RoadAnomaly/images/*.jpg' --config-file '/home/nberardo/mask2anomaly_v2/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml' --output '/home/nberardo/mask2anomaly_v2/output/result_eval/vicreg/'
