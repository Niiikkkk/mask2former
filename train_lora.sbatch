#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2f_lora
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/train_lora.log
#SBATCH --error=/home/nberardo/mask2former/train_lora.err
###### 1 Load the module
module load nvidia/cudasdk/11.6
#module load intel/python/3
## 2 Activate the virtual environment
#source /home/nberardo/myenv/bin/activate

source /home/nberardo/miniconda/bin/activate
conda activate my_env
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#convert pth weights to pkl
#python tools/convert-torchvision-to-d2.py /home/nberardo/simsiam/resnet50.pth /home/nberardo/mask2anomaly_v2/backbone_weights/simsiam_resnet50.pkl

python code_for_stuff.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml \
  --num-gpus 1

#python evaluation_ood_lora.py --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_lora.yaml