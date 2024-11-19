#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=ray_train
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/ft_all.log
#SBATCH --error=/home/nberardo/mask2former/ft_all.err
###### 1 Load the module
module load nvidia/cudasdk
# 2 Activate the virtual environment
source /home/nberardo/miniconda/bin/activate
conda activate my_env
# 3 Run the python script

export DETECTRON2_DATASETS=/home/nberardo/Datasets
export CUDA_HOME=/home/nberardo/cuda_11.8

#python train_lp_ft.py \
#  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
#  --num-gpus 2

#python register_RA.py

python fine_tune_all.py \
      --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_bt.yaml \
      --num-gpus 2