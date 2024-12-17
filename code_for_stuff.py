import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import torch
import tqdm
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger

from evaluation_on_ood import func

from component_metric import get_threshold_from_PRC
from mask2former import add_maskformer2_config


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
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def print_img(image_to_plot,path_to_save):
    print(image_to_plot.shape, path_to_save)
    sys.stdout.flush()
    plt.axis("off")
    plt.tight_layout()
    if "image" in path_to_save or "label" in path_to_save:
        plt.imshow(cv2.cvtColor(image_to_plot, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(decode_segmap(image_to_plot[0]))
    plt.savefig(path_to_save)
    plt.clf()

def draw_prediction(model, img_paths, img_out, ssl_name):
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
        pathGT = pathGT.replace("gtFine.png", "gtFine_color.png")
        label = read_image(pathGT, format="BGR")
        save_label_path = os.path.join(img_out, ssl_name, "label_" + str(num) + ".png")
        print_img(label,save_label_path)


if __name__ == "__main__":
    import re
    import pandas as pd

    # Input text
    log_text = """
    ==> output/train/bt_FT_1k_5e-5_all/log.txt <==
[12/15 17:32:50] d2.evaluation.testing INFO: copypaste: 75.4052,59.5973,89.9280,78.8584

==> output/train/bt_FT_2k_5e-5_all/log.txt <==
[12/15 23:26:55] d2.evaluation.testing INFO: copypaste: 76.2235,59.9747,90.0530,78.9100

==> output/train/bt_FT_3k_5e-5_all/log.txt <==
[12/17 08:59:08] d2.evaluation.testing INFO: copypaste: 76.1618,59.4823,90.1587,78.6892

==> output/train/bt_FT_4k_5e-5_all/log.txt <==
[12/09 23:54:26] d2.evaluation.testing INFO: copypaste: 76.6192,60.1636,90.2520,78.9513

==> output/train/bt_FT_6k_5e-5_all/log.txt <==
[12/10 00:32:13] d2.evaluation.testing INFO: copypaste: 77.0318,60.8969,90.3209,79.1844

==> output/train/bt_FT_8k_5e-5_all/log.txt <==
[12/11 11:18:18] d2.evaluation.testing INFO: copypaste: 76.6515,59.8754,90.3702,79.1428
[mask2former]nberardo@legionlogin$ tail -n 1 output/train/bt_FT_*_all/log.txt
==> output/train/bt_FT_10k_6e-5_all/log.txt <==
[12/15 14:21:19] d2.evaluation.testing INFO: copypaste: 76.9926,60.3748,90.3635,79.2698

==> output/train/bt_FT_10k_8e-5_all/log.txt <==
[12/15 19:21:31] d2.evaluation.testing INFO: copypaste: 76.7753,60.7187,90.4986,79.5119

==> output/train/bt_FT_1k_5e-5_all/log.txt <==
[12/15 17:32:50] d2.evaluation.testing INFO: copypaste: 75.4052,59.5973,89.9280,78.8584

==> output/train/bt_FT_1k_8e-5_all/log.txt <==
[12/15 16:51:51] d2.evaluation.testing INFO: copypaste: 76.2154,59.5658,89.9708,78.7960

==> output/train/bt_FT_2k_5e-5_all/log.txt <==
[12/15 23:26:55] d2.evaluation.testing INFO: copypaste: 76.2235,59.9747,90.0530,78.9100

==> output/train/bt_FT_2k_8e-5_all/log.txt <==
[12/10 22:45:17] d2.evaluation.testing INFO: copypaste: 75.8184,59.3647,90.0933,79.1881

==> output/train/bt_FT_3k_5e-5_all/log.txt <==
[12/17 08:59:08] d2.evaluation.testing INFO: copypaste: 76.1618,59.4823,90.1587,78.6892

==> output/train/bt_FT_3k_7e-5_all/log.txt <==
[12/11 15:18:36] d2.evaluation.testing INFO: copypaste: 76.8083,60.5168,90.2169,79.0699

==> output/train/bt_FT_4k_1e-5_all/log.txt <==
[12/10 09:22:41] d2.evaluation.testing INFO: copypaste: 75.4533,59.1702,89.7927,78.3116

==> output/train/bt_FT_4k_5e-5_all/log.txt <==
[12/09 23:54:26] d2.evaluation.testing INFO: copypaste: 76.6192,60.1636,90.2520,78.9513

==> output/train/bt_FT_4k_7e-5_all/log.txt <==
[12/10 10:35:45] d2.evaluation.testing INFO: copypaste: 77.1114,60.7443,90.3433,79.3092

==> output/train/bt_FT_4k_8e-5_all/log.txt <==
[12/11 00:01:16] d2.evaluation.testing INFO: copypaste: 77.1644,60.8179,90.3321,78.8898

==> output/train/bt_FT_5k_6e-5_all/log.txt <==
[12/14 19:55:32] d2.evaluation.testing INFO: copypaste: 76.8534,60.0818,90.3595,79.1145

==> output/train/bt_FT_5k_8e-5_all/log.txt <==
[12/14 10:56:56] d2.evaluation.testing INFO: copypaste: 77.0594,60.5294,90.4127,79.2953

==> output/train/bt_FT_6k_1e-5_all/log.txt <==
[12/10 09:53:13] d2.evaluation.testing INFO: copypaste: 76.0060,60.0646,89.9362,78.4674

==> output/train/bt_FT_6k_5e-5_all/log.txt <==
[12/10 00:32:13] d2.evaluation.testing INFO: copypaste: 77.0318,60.8969,90.3209,79.1844

==> output/train/bt_FT_6k_7e-5_all/log.txt <==
[12/13 15:15:25] d2.evaluation.testing INFO: copypaste: 75.9201,59.8979,90.3266,78.9301

==> output/train/bt_FT_6k_8e-5_all/log.txt <==
[12/11 01:45:08] d2.evaluation.testing INFO: copypaste: 77.0464,60.1364,90.3529,79.0557

==> output/train/bt_FT_8k_3e-5_all/log.txt <==
[12/11 14:11:32] d2.evaluation.testing INFO: copypaste: 76.4949,59.5953,90.1895,79.0693

==> output/train/bt_FT_8k_5e-5_all/log.txt <==
[12/11 11:18:18] d2.evaluation.testing INFO: copypaste: 76.6515,59.8754,90.3702,79.1428
    """

    # Regular expression to extract iter, lr, FT, and mIoU
    pattern = r"==> output/train/bt_FT_(\d+k)_(\d+e-?\d+)_(\w+)/log.txt <==\n.*?copypaste: (\d+\.\d+)"

    # Extract matches
    matches = re.findall(pattern, log_text)

    # Format data for the DataFrame
    data = []
    for iter_, lr, ft, miou in matches:
        data.append((iter_, lr, ft, float(miou)))

    # Create DataFrame with separate columns
    df = pd.DataFrame(data, columns=["Iteration", "Learning Rate", "FT", "mIoU"])

    # Save to Excel
    df.to_excel("results_split.xlsx", index=False)
    print("Excel file 'results_split.xlsx' has been created successfully.")

    # args = parse_args()
    # cfg = setup_cfgs(args)
    # logger = setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    # cfg.defrost()
    # models = [#"simsiam", "bt", "bt-down-freezed", "bt-freezed", "dino",
    #           #"dino-down-freezed", "dino-freezed", "moco-v1",
    #           #"moco-v1-freezed_NEW",
    #           #"moco_v1_downloaded", "moco-v2", "moco-v2-freezed_NEW", "moco_v2_downloaded",
    #           #"simsiam_freezed",
    #           #"vicreg", "vicreg_down_freeze", "vicreg-freezed"
    #           "no_pre"]
    # for model_name in models:
    #     cfg.MODEL.WEIGHTS = os.path.join("/home/nberardo/mask2former/output/train", model_name, "model_final.pth")
    #     cfg.OUTPUT_DIR = os.path.join("/home/nberardo/mask2former/results", model_name)
    #     model = DefaultPredictor(cfg)
    #     func(model,args,cfg)
    # for i in range(30):
    #     x = np.asarray(Image.open(f"/Users/nicholas.berardo/Desktop/fs_static/labels_masks/{i}.png"))
    #     print(np.unique(x))
    # args = parse_args()
    # cfg = setup_cfgs(args)
    # predictor = DefaultPredictor(cfg)
    # # img = read_image("/Users/nicholas.berardo/Desktop/FS_LostFound_full/images/1.png", format="BGR")
    # for i in range(100):
    #     gt = np.asarray(Image.open(f"/Users/nicholas.berardo/Desktop/FS_LostFound_full/labels_masks/{i}.png"))
    #     print(np.unique(gt))
    # pred = predictor(img)["sem_seg"].unsqueeze(0)
    #
    # th = get_threshold_from_PRC(1-torch.max(pred,dim=1)[0], np.expand_dims(gt,0))
    # print(th)
    #
    # plt.imshow(torch.max(pred.squeeze(),dim=0)[1])
    # plt.show()