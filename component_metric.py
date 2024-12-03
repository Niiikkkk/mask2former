#Here the sIoU for ANOMALIES, not for ID classes.

import numpy as np
from scipy.ndimage.measurements import label
from easydict import EasyDict
from sklearn.metrics import precision_recall_curve

def get_threshold_from_PRC(anomaly_p: np.ndarray, label_pixel_gt: np.ndarray):
    """
    function that computes the threshold for the anomaly score based on the precision-recall curve
    anomaly_p: (numpy array) anomaly score prediction
    label_pixel_gt: (numpy array) ground truth anomaly mask
    """

    """compute precision-recall curve"""

    n_ = anomaly_p.shape[0]
    thresholds_array = []

    for i in range(n_):
        print(np.unique(label_pixel_gt[i]))

        prec, rec, thresholds = precision_recall_curve(label_pixel_gt[i][label_pixel_gt[i] != 255],
                                                       anomaly_p[i][label_pixel_gt[i] != 255])

        f1_scores = (2 * prec * rec) / (prec + rec)
        idx = np.nanargmax(f1_scores)
        if len(thresholds) == 1:
            thresholds_array.append(thresholds[0])
        else:
            thresholds_array.append(thresholds[idx])

    return thresholds_array

def default_instancer(anomaly_p: np.ndarray, label_pixel_gt: np.ndarray, thresh_p: float,
                      thresh_segsize: int, thresh_instsize: int = 0):
    """
    anomaly_p: (numpy array) anomaly score prediction
    label_pixel_gt: (numpy array) ground truth anomaly mask
    thresh_p: (float) threshold for anomaly score
    thresh_segsize: (int) threshold for connected component size. Component (related to the segmentation) smaller that this value are discarded (value in pixel)
    thresh_instsize: (int) threshold for instance size. Component (related to the ground truth) smaller that this value are discarded (value in pixel)
    """

    """segmentation from pixel-wise anoamly scores"""
    segmentation = np.copy(anomaly_p)
    segmentation[anomaly_p > thresh_p] = 1
    segmentation[anomaly_p <= thresh_p] = 0

    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.uint8)
    """
    The structure is a 3x3 matrix with all one. 
    anomaly_gt and anomaly_pred are binary images, with 1 for the anomaly and 0 for the background.
    The label function allows to find the number of element of 1 (anomaly) in the image and locate them, it does so by looking 
        at the 8 neighbors of each pixel (this is given by the structure).
        label returns the number of instances, and an array that have values for each instance + background.
        So if I have 2 cows (anomalies in the GT) I will have 2 instances and the array will have 3 values: 0 for the background, 1 for the first cow and 2 for the second cow.
    So if in the label I have 3 OOD object (so they have value 1), then label with identify 3 object in the image, same thing for the prediction.
    """
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove connected components below size threshold for both GT and pred"""
    if thresh_segsize is not None:
        minimum_cc_sum  = thresh_segsize
        labeled_mask = np.copy(anomaly_seg_pred)
        for comp in range(n_seg_pred+1):
            if len(anomaly_seg_pred[labeled_mask == comp]) < minimum_cc_sum:
                anomaly_seg_pred[labeled_mask == comp] = 0
    labeled_mask = np.copy(anomaly_instances)
    label_pixel_gt = label_pixel_gt.copy() # copy for editing
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    """restrict to region of interest"""
    mask_roi = label_pixel_gt < 255
    segmentation_filtered = np.copy(anomaly_seg_pred).astype("uint8")
    segmentation_filtered[anomaly_seg_pred>0] = 1
    segmentation_filtered[mask_roi==255] = 0

    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi], segmentation_filtered, anomaly_instances, anomaly_seg_pred


def anomaly_instances_from_mask(segmentation: np.ndarray, label_pixel_gt: np.ndarray, thresh_instsize: int = 0):
    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.uint8)
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove ground truth connected components below size threshold"""
    labeled_mask = np.copy(anomaly_instances)
    label_pixel_gt = label_pixel_gt.copy() # copy for editing
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    """restrict to region of interest"""
    mask_roi = label_pixel_gt < 255
    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi]


def segment_metrics(anomaly_instances, anomaly_seg_pred, iou_thresholds=np.linspace(0.25, 0.75, 11, endpoint=True)):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_instances: (numpy array) anomaly instance annoation
    anomaly_seg_pred: (numpy array) anomaly instance prediction
    iou_threshold: (float) threshold for true positive
    return: (dictionary) results containing those main things:
        - the number of true positive, false negative and false positive for each IoU threshold
        - sIoU_gt: the adjusted IoU for the ground truth instances
        - sIoU_pred: the adjusted IoU for the prediction instances
    """

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []

    for i in np.unique(anomaly_instances[anomaly_instances>0]):
        """i is a compoment or instance of the ground truth"""
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        """tp_loc are the pixels related to the component i"""
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        """calc area of intersection"""
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])

        adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(
            anomaly_instances == i) - intersection - adjustment
        sIoU_gt.append(intersection / adjusted_union)
        size_gt.append(np.sum(anomaly_instances == i))

    """Loop over prediction instances"""
    sIoU_pred = []
    size_pred = []


    """
    prec_pred is the precision or PPV
    """
    prec_pred = []
    for i in np.unique(anomaly_seg_pred[anomaly_seg_pred>0]):
        tp_loc = anomaly_instances[anomaly_seg_pred == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
        adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(
            anomaly_seg_pred == i) - intersection - adjustment
        sIoU_pred.append(intersection / adjusted_union)
        size_pred.append(np.sum(anomaly_seg_pred == i))
        prec_pred.append(intersection / np.sum(anomaly_seg_pred == i))

    sIoU_gt = np.array(sIoU_gt)
    sIoU_pred = np.array(sIoU_pred)
    size_gt = np.array((size_gt))
    size_pred = np.array(size_pred)
    prec_pred = np.array(prec_pred)

    """create results dictionary"""
    results = EasyDict(sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred, prec_pred=prec_pred)
    for t in iou_thresholds:
        results["tp_" + str(int(t*100))] = np.count_nonzero(sIoU_gt >= t)
        results["fn_" + str(int(t*100))] = np.count_nonzero(sIoU_gt < t)
        # results["fp_" + str(int(t*100))] = np.count_nonzero(sIoU_pred < t)
        results["fp_" + str(int(t*100))] = np.count_nonzero(prec_pred < t)

    return results

def aggregate(frame_results: list, thresh_sIoU=np.linspace(0.25, 0.75, 11, endpoint=True)):

        sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
        sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)
        prec_pred_mean = sum(np.sum(r.prec_pred) for r in frame_results) / sum(len(r.prec_pred) for r in frame_results)
        ag_results = {"tp_mean" : 0., "fn_mean" : 0., "fp_mean" : 0., "f1_mean" : 0.,
                      "sIoU_gt" : sIoU_gt_mean, "sIoU_pred" : sIoU_pred_mean, "prec_pred": prec_pred_mean}
        print("Mean sIoU GT   :", sIoU_gt_mean)
        print("Mean sIoU PRED :", sIoU_pred_mean)
        print("Mean Precision PRED :", prec_pred_mean)
        for t in thresh_sIoU:
            tp = sum(r["tp_" + str(int(t*100))] for r in frame_results)
            fn = sum(r["fn_" + str(int(t*100))] for r in frame_results)
            fp = sum(r["fp_" + str(int(t*100))] for r in frame_results)
            f1 = (2 * tp) / (2 * tp + fn + fp)
            if t in [0.25, 0.50, 0.75]:
                ag_results["tp_" + str(int(t * 100))] = tp
                ag_results["fn_" + str(int(t * 100))] = fn
                ag_results["fp_" + str(int(t * 100))] = fp
                ag_results["f1_" + str(int(t * 100))] = f1
            # print("---sIoU thresh =", t)
            # print("Number of TPs  :", tp)
            # print("Number of FNs  :", fn)
            # print("Number of FPs  :", fp)
            # print("F1 score       :", f1)
            ag_results["tp_mean"] += tp
            ag_results["fn_mean"] += fn
            ag_results["fp_mean"] += fp
            ag_results["f1_mean"] += f1

        ag_results["tp_mean"] /= len(thresh_sIoU)
        ag_results["fn_mean"] /= len(thresh_sIoU)
        ag_results["fp_mean"] /= len(thresh_sIoU)
        ag_results["f1_mean"] /= len(thresh_sIoU)
        print("---sIoU thresh averaged")
        print("Number of TPs  :", ag_results["tp_mean"])
        print("Number of FNs  :", ag_results["fn_mean"])
        print("Number of FPs  :", ag_results["fp_mean"])
        print("F1 score       :", ag_results["f1_mean"])

        return ag_results