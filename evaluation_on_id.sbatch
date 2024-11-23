#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=m2f_tests
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/mask2former/eval_on_id.log
#SBATCH --error=/home/nberardo/mask2former/eval_on_id.err
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

#python train_lp_ft.py \
#  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
#  --num-gpus 2

#python register_RA.py

python evaluation_on_id.py --config-file /home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference_id.yaml