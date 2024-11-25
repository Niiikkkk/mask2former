import argparse

import matplotlib.pyplot as plt
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, DefaultPredictor

from train_net import setup, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Stuff')
    parser.add_argument("--config_file",
        default="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_TMP.yaml",
        metavar="FILE",
        help="path to config file",)
    parser.add_argument("--input",
        default="/Users/nicholas.berardo/Desktop/RoadAnomaly/images/*.jpg",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",)

    args = parser.parse_args()
    return args

def draw_prediction(model, image, output):
    prediction = model(image)["sem_seg"]
    out = torch.max(prediction,dim=1)[1].detach().cpu().numpy()
    plt.imshow(out)
    plt.savefig(output)


if __name__ == "__main__":
    args = parse_args()
    cfg = setup(args)
    model = DefaultPredictor(cfg)
    draw_prediction(model, args.input, args.output)
