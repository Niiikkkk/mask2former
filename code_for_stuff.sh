#!/bin/bash

python code_for_stuff.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --input \
  /home/nberardo/Datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_012038_leftImg8bit.png \
  --output /home/nberardo/mask2former/prediction_example

