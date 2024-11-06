#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2f_dino
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/train_dino.log
#SBATCH --error=/home/nberardo/mask2former/train_dino.err
###### 1 Load the module
module load nvidia/cudasdk
# 2 Activate the virtual environment
source /home/nberardo/miniconda/bin/activate
conda activate venv
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

cd mask2former/modeling/pixel_decoder/ops
sh make.sh

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/simsiam/resnet50.pth /home/nberardo/mask2anomaly_v2/backbone_weights/simsiam_resnet50.pkl

#python tools/convert-pretrained-model-to-d2.py /home/nberardo/mask2former/backbone_weights/swin_s-5e29d889.pth /home/nberardo/mask2former/backbone_weights/swin_s-5e29d889.pkl

#python train_net.py \
#  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_dino.yaml \
#  --num-gpus 2
