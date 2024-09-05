import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config

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

cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml")
cfg.freeze()
print(cfg)
model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)
#model = torch.load("../../../legion/model_0079999.pth", weights_only=False, map_location='cpu')["model"]
img = read_image("/home/nberardo/Datasets/FS_LostFound_full/images/54.png")
#print(model.state_dict().keys())
res = model(img)
print(res.shape)
