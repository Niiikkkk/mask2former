import argparse
import os.path

import matplotlib.pyplot as plt
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import numpy as np

from mask2former import add_maskformer2_config
from register_RA import label_colours


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

colors = [
        #[  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

# makes a dictionary with key:value. For example 0:[128, 64, 128]
label_colours = dict(zip(range(19), colors))

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
    print(temp.shape)
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def print_img(image_to_plot,path_to_save):
    if "image" in path_to_save:
        plt.imshow(cv2.cvtColor(image_to_plot, cv2.COLOR_BGR2RGB))
    else:
        if "label" in path_to_save:
            image_to_plot = np.transpose(image_to_plot,(2,0,1))
        plt.imshow(decode_segmap(image_to_plot[0]))
    plt.savefig(path_to_save)
    plt.clf()

def draw_prediction(model, img_paths, img_out, ssl_name):
    print(img_paths)
    os.makedirs(os.path.join(img_out, ssl_name),exist_ok=True)
    for num,img_path in enumerate(tqdm.tqdm(img_paths)):
        image = read_image(img_path, format="BGR")
        prediction = model(image)["sem_seg"].unsqueeze(0)
        prediction_img = torch.max(prediction,dim=1)[1].detach().cpu().numpy()
        save_prediction_path = os.path.join(img_out, ssl_name, "prediction_" + str(num) + ".png")
        print_img(prediction_img,save_prediction_path)
        save_image_path = os.path.join(img_out, ssl_name, "image_" + str(num) + ".png")
        print_img(image,save_image_path)
        pathGT = img_path.replace("leftImg8bit", "gtFine")
        pathGT = pathGT.replace("gtFine.png", "gtFine_labelTrainIds.png")
        label = read_image(pathGT, format="BGR")
        save_label_path = os.path.join(img_out, ssl_name, "label_" + str(num) + ".png")
        print_img(label,save_label_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = setup_cfgs(args)
    model = DefaultPredictor(cfg)
    ssl_name = cfg.MODEL.WEIGHTS.split('/')[-2]
    draw_prediction(model, args.input, args.output, ssl_name)
