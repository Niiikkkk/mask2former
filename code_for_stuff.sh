#!/bin/bash

python code_for_stuff.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --input \
  /home/nberardo/Datasets/cityscapes/leftImg8bit/test/berlin/berlin_000100_000019_leftImg8bit.png \
  --output /home/nberardo/mask2former/prediction_example

#  /home/nberardo/Datasets/cityscapes/leftImg8bit/test/bielefeld/bielefeld_000000_012080_leftImg8bit.png \
#  /home/nberardo/Datasets/cityscapes/leftImg8bit/test/bonn/bonn_000010_000019_leftImg8bit.png \
#  /home/nberardo//Datasets/cityscapes/leftImg8bit/test/leverkusen/leverkusen_000010_000019_leftImg8bit.png \
#  /home/nberardo//Datasets/cityscapes/leftImg8bit/test/mainz/mainz_000001_010853_leftImg8bit.png \
#  /home/nberardo//Datasets/cityscapes/leftImg8bit/test/munich/munich_000110_000019_leftImg8bit.png \

