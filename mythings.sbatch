#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2f_tests
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/mythings.log
#SBATCH --error=/home/nberardo/mask2former/mythings.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

python train_lp_ft.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --num-gpus 2

#python register_RA.py

#python evaluation_on_ood.py --input /home/nberardo/Datasets/RoadAnomaly21/images/*.png \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml
#
#python evaluation_on_ood.py --input /home/nberardo/Datasets/RoadAnomaly/images/*.jpg \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml
#
#python evaluation_on_ood.py --input /home/nberardo/Datasets/RoadObsticle21/images/*.webp \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml
#
#python evaluation_on_ood.py --input /home/nberardo/Datasets/FS_LostFound_full/images/*.png \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml
#
#python evaluation_on_ood.py --input /home/nberardo/Datasets/fs_static/images/*.jpg \
# --config_file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml