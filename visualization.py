import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import color_sequences
import matplotlib


def visualize_anomlay_over_img(img:np.ndarray, anomaly_pred: np.ndarray, threshold:float, label:np.ndarray = None,
                               path_to_save:str = None):
    """
    Visualize the anomaly mask over the image
    Args:
        img (np.ndarray): The image
        anomaly_pred (np.ndarray): The anomaly mask
        threshold (float): The threshold to say "it's anomaly"
        label (np.ndarray, optional): The label of the anomaly. Defaults to None.
        path_to_save (str, optional): The path to save the image. Defaults to None.
    Returns:
        np.ndarray: The image with the anomaly
    """
    anomaly_mask = np.zeros(anomaly_pred.shape)
    anomaly_mask[anomaly_pred > threshold] = 1
    anomaly_mask[anomaly_pred <= threshold] = 0
    if label is not None:
        label_mask = np.zeros(label.shape)
        label_mask[label == 1] = 1

    binary_cmap = ListedColormap(["black","red"])

    if label is not None:
        plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(anomaly_mask ,alpha=0.7, cmap=binary_cmap, vmin=0, vmax=1)

    if label is not None:
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(label_mask ,alpha=0.7, cmap=binary_cmap, vmin=0, vmax=1)
    plt.show()
    if path_to_save:
        plt.savefig(path_to_save)

def visualize_instances_over_img(img:np.ndarray, instances:np.ndarray):
    """
    Visualize the instances over the image
    Args:
        img (np.ndarray): The image
        instances (np.ndarray): The instances
    Returns:
        np.ndarray: The image with the instances
    """
    num_colors = np.unique(instances)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(instances, cmap="magma", alpha=0.7, vmin=0, vmax=len(num_colors)-1)
    plt.show()

