import argparse

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_setup, launch, DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from mask2former import add_maskformer2_config
import os
import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from ood_metrics import fpr_at_95_tpr, plot_pr
from component_metric import segment_metrics, anomaly_instances_from_mask, aggregate, get_threshold_from_PRC, \
    default_instancer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation on OOD')
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
        default="../../results",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",)
    parser.add_argument("--weights",
                        help="Path to a file with model weights")

    args = parser.parse_args()
    return args

def setup_cfgs(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    # if args.output:
    #     cfg.OUTPUT_DIR = args.output
    # if args.weights:
    #     cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg

def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def func():
    args = parse_args()
    cfg = setup_cfgs(args)
    logger = setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    logger.info("Arguments: " + str(args))

    # model = DefaultTrainer.build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
    # model.eval()


    model = DefaultPredictor(cfg)

    file_path = os.path.join(cfg.OUTPUT_DIR, 'results.txt')

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    file = open(file_path, 'a')

    file.write(args.input[0].split('/')[4] + "\n")

    predictions = np.array([])
    gts = np.array([])
    results = np.array([])

    for num, img_path in enumerate(tqdm.tqdm(args.input)):
        with torch.no_grad():
            img = read_image(img_path, format="BGR")
            # height, width = img.shape[:2]
            # img = torch.as_tensor(img.astype(np.float32).transpose(2, 0, 1))
            # input = [{"image": img, "height": height, "width": width}]
            # prediction = model(input)[0]["sem_seg"].unsqueeze(0)  # Here C = 19, cityscapes classes

            prediction = model(img)["sem_seg"].unsqueeze(0)

            # if num == 0:
            #     print(prediction.squeeze())

            # if num == 0:
            #     out_img = torch.max(prediction.squeeze(),axis=0)[1].detach().cpu().numpy()
            #     plt.imshow(out_img)
            #     plt.show()
            prediction_ = 1 - torch.max(prediction, axis=1)[0]

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
                # RA21 has label 2 for anomaly, but we want it to be 1, so change it
                ood_gts = np.where((ood_gts == 2), 1, ood_gts)

            # Ignore the "void" label, that is 255
            # 0 => In distrubiton
            # 1 => Out of distribution
            # 255 => Void, so ignore it

            prediction_ = prediction_.detach().cpu().numpy().squeeze().squeeze()
            prediction_ = np.expand_dims(prediction_, 0).astype(np.float32)
            ood_gts = np.expand_dims(ood_gts, 0)

            # compute component level metric
            # get the threshold in order to say "it's anomaly"
            threshold_to_anomaly = get_threshold_from_PRC(prediction_[ood_gts != 255].squeeze(),
                                                          ood_gts[ood_gts != 255].squeeze())
            # get the instances of the anomaly and gt
            anomaly_instances, anomaly_seg_pred, _, anomaly_instances_for_vis, anomaly_seg_pred_for_vis = default_instancer(
                prediction_.squeeze(), ood_gts.squeeze(), threshold_to_anomaly, 1000, 100)
            # get the metrics
            result = segment_metrics(anomaly_instances, anomaly_seg_pred)
            results = np.append(results, result)

            # Visualize anomaly over the img
            # visualize_anomlay_over_img(img, prediction_.squeeze(), threshold_to_anomaly)
            # visualize_instances_over_img(img,anomaly_seg_pred_for_vis)
            predictions = np.append(predictions, prediction_)
            gts = np.append(gts, ood_gts)

    # Eval...

    final_res = aggregate(results)
    predictions = predictions[(gts != 255)]
    gts = gts[(gts != 255)]
    fpr, tpr, threshold = roc_curve(gts, predictions)
    roc_auc = auc(fpr, tpr)
    fpr = fpr_at_95_tpr(predictions, gts)
    ap = average_precision_score(gts, predictions)

    res = {}
    res["AUROC"] = roc_auc
    res["FPR@TPR95"] = fpr
    res["AUPRC"] = ap

    file.write(
        "AUROC: " + str(res["AUROC"]) +
        " FPR@TPR95: " + str(res["FPR@TPR95"]) +
        " AUPRC: " + str(res["AUPRC"]) +
        " sIoU_pred: " + str(final_res["prec_pred"]) + "\n")

if __name__=="__main__":
    func()
