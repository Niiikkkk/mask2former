#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=m2f_no_pre
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/train_no_pre.log
#SBATCH --error=/home/nberardo/mask2former/train_no_pre.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/simsiam/resnet50_downloaded.pth /home/nberardo/mask2former/backbone_weights/simsiam_resnet50_downloaded.pkl

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_no_pre.yaml \
  --num-gpus 2
