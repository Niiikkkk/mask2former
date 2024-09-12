import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
import detectron2.utils.comm as comm
import argparse
from matplotlib import pyplot as plt

"""x = cv2.imread("../../../legion/Datasets/FS_LostFound_full/labels_masks/54.png")
# RA21 -> 0:Cityscapes 1:Anomaly 255:VOID
# RO21 -> 0:Street 1:Anomaly 255:Other things
# RO   -> 0:Ciryscapes 2:Anomaly
# FS_S -> 0:Cityscapes 1:Anomaly 255:VOID
# FS_L&F -> 0:Cityscapes 1:Anomaly 255:VOID
print(np.unique(x))
x = np.where((x==0), 0, x)
x = np.where((x==1), 100, x)
x = np.where((x==255), 255, x)
cv2.imshow("image",x)
cv2.waitKey(0)cd 
cv2.destroyAllWindows()"""


colors = [
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


label_colours = dict(zip(range(19), colors))


def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[1], temp.shape[2], 3))
    print(temp.shape)
    print(rgb.shape)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    print(rgb.shape)
    return rgb


def func() :
    cfg = get_cfg()
# for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("../configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_TMP.yaml")
    cfg.freeze()

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(cfg))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    #read_image return HWC
    img = read_image("/Users/nicholas.berardo/Desktop/RoadAnomaly/images/0.jpg", format="BGR")

    height, width = img.shape[:2]


    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))


    input = [{"image": img, "height": height, "width": width}]
    model.training = False
    res = model(input)[0]["sem_seg"]

    res = torch.max(res,axis=0)
    res = res[1]

    im = res.detach().cpu().numpy()
    plt.imshow(im)
    plt.show()

if __name__ == "__main__":
    func()