#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=m2f_moco2
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/train_moco2.log
#SBATCH --error=/home/nberardo/mask2former/train_moco2.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/moco/output_dir/moco_v2_r50/resnet50.pth /home/nberardo/mask2former/backbone_weights/moco_v2_resnet50.pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/moco/output_dir/moco_v1_r50/resnet50.pth /home/nberardo/mask2former/backbone_weights/moco_v1_resnet50.pkl

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --num-gpus 2
