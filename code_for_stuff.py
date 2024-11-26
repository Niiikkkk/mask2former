import argparse
import matplotlib.pyplot as plt
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
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
    parser.add_argument("--output",
        default="/Users/nicholas.berardo/Desktop/RoadAnomaly/output.jpg",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",)

    args = parser.parse_args()
    return args

def setup_cfgs(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    # if args.output:
    #     cfg.OUTPUT_DIR = args.output
    # if args.weights:
    #     cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg

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
