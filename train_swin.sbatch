#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=m2f_swin
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/train_swin.log
#SBATCH --error=/home/nberardo/mask2former/train_swin.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-pretrained-swin-model-to-d2.py /home/nberardo/mask2former/backbone_weights/swin_small_patch4_window7_224.pth /home/nberardo/mask2former/backbone_weights/swin_small_patch4_window7_224.pkl

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml \
  --num-gpus 2