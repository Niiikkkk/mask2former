import argparse
import os.path

import matplotlib.pyplot as plt
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser, DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from tensorboardX.summary import image

from mask2former import add_maskformer2_config
from train_net import setup, Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Stuff')
    parser.add_argument("--config-file",
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

def print_img(image_to_plot,path_to_save):
    plt.imshow(image_to_plot)
    plt.savefig(path_to_save)
    plt.close()

def draw_prediction(model, img_paths, img_out, ssl_name):
    for num,img_path in enumerate(img_paths):
        image = read_image(img_path, format="BGR")
        prediction = model(image)["sem_seg"]
        prediction_img = torch.max(prediction,dim=1)[1].detach().cpu().numpy()
        save_prediction_path = os.path.join(img_out, ssl_name, "prediction_" + str(num) + ".png")
        print(save_prediction_path)
        print_img(prediction_img,save_prediction_path)
        save_image_path = os.path.join(img_out, ssl_name, "image_" + str(num) + ".png")
        print(save_image_path)
        print_img(image,save_image_path)
        pathGT = img_path.replace("leftImg8bit", "gtFine")
        pathGT = pathGT.replace("gtFine.png", "gtFine_color.png")
        label = read_image(pathGT, format="BGR")
        save_label_path = os.path.join(img_out, ssl_name, "label_" + str(num) + ".png")
        print(save_label_path)
        print_img(label,save_label_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = setup_cfgs(args)
    model = DefaultPredictor(cfg)
    ssl_name = cfg.MODEL.WEIGHTS.split('/')[-2]
    draw_prediction(model, args.input, args.output, ssl_name)
