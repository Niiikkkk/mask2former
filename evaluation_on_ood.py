import argparse
import sys

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_setup, launch, DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from ood_metrics import fpr_at_95_tpr, plot_pr
from component_metric import segment_metrics, anomaly_instances_from_mask, aggregate, get_threshold_from_PRC, \
    default_instancer
from uncertainty_metric import prediction_rejection_ratio
from visualization import visualize_anomlay_over_img


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

def plot_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def func(model, args, cfg):

    file_path = os.path.join(cfg.OUTPUT_DIR, 'results.txt')
    stderr_file = os.path.join(cfg.OUTPUT_DIR, 'stderr.txt')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(stderr_file):
        open(stderr_file, 'w').close()
    sys.stderr = open(stderr_file, 'a')

    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    file = open(file_path, 'a')

    db_name = str(args.input[0].split('/')[4])
    file.write(db_name + "\n")

    predictions = []
    gts = []
    results = np.array([])

    num_images = len(args.input)
    num_image_to_print = np.random.randint(num_images)

    print("Processing images...")

    for num, img_path in enumerate(tqdm(args.input)):
        with torch.no_grad():
            img = read_image(img_path, format="BGR")
            # height, width = img.shape[:2]
            # img = torch.as_tensor(img.astype(np.float32).transpose(2, 0, 1))
            # input = [{"image": img, "height": height, "width": width}]
            # prediction = model(input)[0]["sem_seg"].unsqueeze(0)  # Here C = 19, cityscapes classes

            prediction = model(img)["sem_seg"].unsqueeze(0)

            # if num == 0:
            #     print(prediction.squeeze())

            #THIS IS MSP 1-max(Scores)
            prediction_ = 1 - torch.max(prediction, dim=1)[0]

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

            #Some labels are just 0 and 255 (like in FS_static), so skip those images
            if 1 not in ood_gts:
                if num_image_to_print == num:
                    num_image_to_print+=1
                continue

            # Ignore the "void" label, that is 255
            # 0 => In distrubiton
            # 1 => Out of distribution
            # 255 => Void, so ignore it

            prediction_ = prediction_.detach().cpu().numpy().squeeze().squeeze()
            prediction_ = np.expand_dims(prediction_, 0).astype(np.float32)
            ood_gts = np.expand_dims(ood_gts, 0)

            # compute component level metric
            # get the threshold in order to say "it's anomaly"
            threshold_to_anomaly = get_threshold_from_PRC(prediction_,
                                                          ood_gts)


            #print anomaly over prediction
            if num == num_image_to_print:
                out_img = torch.max(prediction.squeeze(),dim=0)[1].detach().cpu().numpy()
                plt.axis("off")
                plt.tight_layout()
                # visualize_anomlay_over_img(decode_segmap(out_img), prediction_.squeeze(), threshold_to_anomaly,
                #                            path_to_save=os.path.join(cfg.OUTPUT_DIR, db_name + "_" + str(num) + "_prediction.png"))
                visualize_anomlay_over_img(img, prediction_.squeeze(), threshold_to_anomaly,
                                           path_to_save=os.path.join(cfg.OUTPUT_DIR, db_name + "_" + str(num) + "_img.png")
                                           ,is_bgr=True,label=ood_gts.squeeze())
                plt.clf()

            # get the instances of the anomaly and gt
            seg_size = 500
            gt_size = 100
            if "FS_LostFound_full" in img_path:
                #In LostFound, the objects are smaller, so we need to change the seg_size and gt_size into smaller values, otherwise we cut out
                # everything...
                seg_size = 50
                gt_size = 10
            anomaly_instances, anomaly_seg_pred, _, anomaly_instances_for_vis, anomaly_seg_pred_for_vis = default_instancer(
                prediction_.squeeze(), ood_gts.squeeze(), threshold_to_anomaly, seg_size, gt_size)
            # get the metrics
            result = segment_metrics(anomaly_instances, anomaly_seg_pred)
            results = np.append(results, result)

            # if "FS_LostFound_full" in img_path:
            #     #Plot the prediction
            #     visualize_anomlay_over_img(img, prediction_.squeeze(), threshold_to_anomaly, ood_gts.squeeze(),
            #                                path_to_save=os.path.join(cfg.OUTPUT_DIR, db_name + "_" + str(num) + ".png"))
            # Visualize anomaly over the img
            # visualize_anomlay_over_img(img, prediction_.squeeze(), threshold_to_anomaly)
            # visualize_instances_over_img(img,anomaly_seg_pred_for_vis)
            predictions.append(prediction_)
            gts.append(ood_gts)

    print("Starting evaluation...")
    # Eval...
    predictions = np.concatenate(predictions, axis=0)
    gts = np.concatenate(gts, axis=0)

    print("1")
    sys.stdout.flush()

    threshold_to_anomaly = get_threshold_from_PRC(predictions, gts)

    print("Threshold to anomaly (one for each image): ", threshold_to_anomaly)
    sys.stdout.flush()

    prr = prediction_rejection_ratio(gts, predictions, threshold=threshold_to_anomaly)

    print("3")
    sys.stdout.flush()

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
        " sIoU: " + str(final_res["sIoU_gt"]) +
        " PPV: " + str(final_res["prec_pred"]) +
        " PRR: " + str(prr) + "\n")

    print("Done...")

if __name__=="__main__":
    args = parse_args()
    cfg = setup_cfgs(args)
    logger = setup_logger(name="fvcore", output=cfg.OUTPUT_DIR)
    logger.info("Arguments: " + str(args))
    model = DefaultPredictor(cfg)
    func(model,args,cfg)
