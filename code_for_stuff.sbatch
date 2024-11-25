#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=code_for_stuff
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/code_for_stuff.log
#SBATCH --error=/home/nberardo/mask2former/code_for_stuff.err
###### 1 Load the module
module load nvidia/cudasdk
#module load intel/python/3
# 2 Activate the virtual environment
#source /home/nberardo/myenv/bin/activate
# 3 Run the python script

source /home/nberardo/miniconda/bin/activate
conda activate my_env

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/simsiam/resnet50.pth /home/nberardo/mask2anomaly_v2/backbone_weights/simsiam_resnet50.pkl

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --input /home/nberardo/Datasets/cityscapes/leftImg8bit/test/berlin/berlin_000100_000019_leftImg8bit.png \
  --output /home/nberardo/mask2former/predition_example/simsiam.png
