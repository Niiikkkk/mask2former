import argparse

import matplotlib.pyplot as plt
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
import os
import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on OOD')
    parser.add_argument("--config_file",
        default="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_TMP.yaml",
        metavar="FILE",
        help="path to config file",)
    parser.add_argument("--input",
        default="/Users/nicholas.berardo/Desktop/RoadAnomaly/images/0.jpg /Users/nicholas.berardo/Desktop/RoadAnomaly/images/1.jpg",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",
        default="../../results",
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
    setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)

    model.training = False

    file_path = os.path.join(cfg.OUTPUT_DIR, 'results.txt')

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    file = open(file_path, 'a')

    file.write(args.input[0].split('/')[4] + "\n")

    predictions = []
    gts = []

    for num, img_path in enumerate(tqdm.tqdm(args.input)):
        img = read_image(img_path, format="BGR")
        height, width = img.shape[:2]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        input = [{"image": img, "height": height, "width": width}]
        prediction = model(input)[0]["sem_seg"].unsqueeze(0) #Here C = 19, cityscapes classes
        print(prediction.max())
        prediction_ = torch.max(prediction, axis=1)[0]

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
            #RA21 has label 2 for anomaly, but we want it to be 1, so change it
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        # Ignore the "void" label, that is 255
        # 0 => In distrubiton
        # 1 => Out of distribution
        # 255 => Void, so ignore it


        prediction_ = prediction_.detach().cpu().numpy().astype(np.float32)
        ood_gts = np.expand_dims(ood_gts,0)

        predictions.append(prediction_)
        gts.append(ood_gts)

    #Eval...
    predictions = np.array(predictions)
    gts = np.array(gts)

    #1 is anomaly, so ood
    ood_mask = (gts == 1)
    #0 is in distribution
    ind_mask = (gts == 0)

    ood_predictions = predictions[ood_mask]
    ind_predictions = predictions[ind_mask]

    predictions = np.concatenate([ood_predictions, ind_predictions])
    gts = np.concatenate([np.ones(len(ood_predictions)), np.zeros(len(ind_predictions))])

    fpr, tpr, threshold = roc_curve(gts, predictions)
    roc_auc = auc(fpr, tpr)
    fpr_best = fpr[tpr >= 0.95][0]
    ap = average_precision_score(gts,predictions)

    res = {}
    res["AUROC"] = roc_auc
    res["FPR@TPR95"] = fpr_best
    res["AUPRC"] = ap

    print(res)

    file.write("AUROC: " + str(res["AUROC"]) + " FPR@TPR95: " + str(res["FPR@TPR95"]) + " AUPRC: " + str(res["AUPRC"]) + "\n")