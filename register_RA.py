import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_setup
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
    temp = temp.reshape(temp.shape[2], temp.shape[1], temp.shape[0])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    print(temp.shape)
    print(rgb.shape)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_inference.yaml")
cfg.freeze()

setup_logger(name="fvcore")
logger = setup_logger()

model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)

#read_image return HWC
img = read_image("/home/nberardo/Datasets/FS_LostFound_full/images/54.png", format="BGR")

img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
print(img.shape)
input = [{"image": torch.tensor(img).float(), "height": img.shape[1], "width": img.shape[2]}]
model.training = False
res = model(input)[0]["sem_seg"].unsqueeze(0)
print(res.shape)
res = torch.max(res,axis=1)
res = res[1]
print(res.shape)

im = decode_segmap(res.detach().cpu().numpy())
plt.imshow(img)
plt.savefig("/home/nberardo/mask2former/image_results/img.jpg")

