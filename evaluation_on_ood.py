import argparse

from PIL import Image
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
import detectron2.utils.comm as comm
import glob
import os
from sklearn.metrics import roc_curve, auc, average_precision_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on OOD')
    parser.add_argument("--config_file",
        default="/home/nberardo/mask2former/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml",
        metavar="FILE",
        help="path to config file",)
    parser.add_argument("--input",
        default="/home/nberardo/Datasets/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",
        default="/home/nberardo/mask2former/results/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",)

    args = parser.parse_args()
    return args

def setup_cfgs(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

if __name__=="__main__":
    args = parse_args()
    cfg = setup_cfgs(args)
    setup_logger(output=args.output, distributed_rank=comm.get_rank(), name="mask2former")

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)

    model.training = False

    predctions = []
    gts = []

    for img_path in glob.glob(os.path.expanduser(args.input[0])):
        img = read_image(img_path, format="BGR")
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        input = [{"image": torch.tensor(img).float(), "height": img.shape[1], "width": img.shape[2]}]
        prediction = model(input)[0]["sem_seg"].unsqueeze(0) #Here C = 19, cityscapes classes
        prediction = torch.max(prediction, axis=1)[0]

        pathGT = img_path.replace("images", "labels_masks")

        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            #RA has label 2 for anomaly, but we want it to be 1, so change it
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        # Ignore the "void" label, that is 255
        # 0 => In distrubiton
        # 1 => Out of distribution
        # 255 => Void, so ignore it

        if 255 in ood_gts:
            #If void pexels, remove them
            predctions.append(prediction[ood_gts != 255])
            gts.append(ood_gts[ood_gts != 255])
        else:
            predctions.append(prediction)
            gts.append(ood_gts)

    #Eval...
    predictions = np.concatenate(predctions,0)
    gts = np.concatenate(gts,0)
    print(predictions.shape)
    print(gts.shape)

    fpr, tpr, threshold = roc_curve(gts, predictions)
    roc_auc = auc(fpr, tpr)
    fpr_best = fpr[tpr >= 0.95][0]
    ap = average_precision_score(gts,predictions)

    res = {}
    res["AUROC"] = roc_auc
    res["FPR@TPR95"] = fpr_best
    res["AP"] = ap
    print(res)






