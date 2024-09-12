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

"""
cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml")
cfg.freeze()
print(cfg)
default_setup(cfg, "")
setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)
#model = torch.load("../../../legion/model_0079999.pth", weights_only=False, map_location='cpu')["model"]
#read_image return HWC
img = read_image("/home/nberardo/Datasets/FS_LostFound_full/images/54.png", format="BGR")
#print(model.state_dict().keys())
img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
print(img.shape)
input = [{"image": torch.tensor(img).float(), "height": img.shape[1], "width": img.shape[2]}]
model.training = False
res = model(input)
print(res[0]["sem_seg"].shape)"""

model = torch.load("backbone_weights/simsiam_resnet50.pkl",map_location="cpu")
for n,w in model.items():
    print(n)