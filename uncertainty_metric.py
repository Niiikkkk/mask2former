import numpy as np
from sklearn.metrics import auc
import torch
import matplotlib.pyplot as plt

def prediction_rejection_ratio(labels, logits, pred_labels, threshold):
    # Based on https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py

    # compute area between base_error(1-x) and the rejection curve
    # compute area between base_error(1-x) and the oracle curve
    # take the ratio

    input = np.where((logits <= threshold), 0, pred_labels)
    input = np.where((logits > threshold), 1, input)

    plt.imshow(input.squeeze())
    plt.show()

    # Get class probabilities
    labels = torch.tensor(labels).view(-1)
    probs = torch.tensor(logits).view(-1)
    preds = torch.tensor(input).view(-1)

    print(probs.shape)
    print(preds.shape)
    print(labels.shape)

    # confidence, preds = torch.max(probs, dim=1)  # Take as confidence the probability of the predicted class
    # confidence = 1 - confidence #Anomaly confidence
    confidence = probs
    preds = preds[labels!=255]
    labels = labels[labels!=255]

    # the rejection plots needs to reject to the right the most uncertain/less confident samples
    # if uncertainty metric, high means reject, sort in ascending uncertainty;
    # if confidence metric, low means reject, sort in descending confidence

    sorted_idx = torch.argsort(confidence, descending=True)

    # reverse cumulative errors function (rev = from all to first, instead from first error to all)
    rev_cum_errors = []
    # fraction of data rejected, to compute a certain value of rev_cum_errors
    fraction_data = []

    num_samples = preds.shape[0]

    errors = (labels[sorted_idx] != preds[sorted_idx]).float().numpy()

    print("Errors: ", len(errors[errors==1]))

    rev_cum_errors = np.cumsum(errors) / num_samples
    fraction_data = np.array([float(i + 1) / float(num_samples) * 100.0 for i in range(num_samples)])

    base_error = rev_cum_errors[-1]  # error when all data is taken into account

    # area under the rejection curve (used later to compute area between random and rejection curve)
    auc_uns = 1.0 - auc(fraction_data / 100.0, rev_cum_errors[::-1] / 100.0)

    # random rejection baseline, it's 1 - x line "scaled" and "shifted" to pass through base error and go to 100% rejection
    random_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(num_samples)) for i in range(num_samples)],
        dtype=np.float32)
    # area under random rejection, should be 0.5
    auc_rnd = 1.0 - auc(fraction_data / 100.0, random_rejection / 100.0)

    # oracle curve, the oracle is assumed to commit the base error
    # making the oracle curve commit the base error allows to remove the impact of the base error when computing
    # the ratio of areas
    # line passing through base error at perc_rej = 0, and crossing
    # the line goes from x=0 to x=base_error/100*num_samples <- this is when the line intersects the x axis
    # which means the oracle ONLY REJECTS THE SAMPLES THAT ARE MISCASSIFIED
    # afterwards the function is set to zero
    orc_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(base_error / 100.0 * num_samples)) for i in
         range(int(base_error / 100.0 * num_samples))], dtype=np.float32)
    orc = np.zeros_like(rev_cum_errors)
    orc[0:orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1.0 - auc(fraction_data / 100.0, orc / 100.0)

    # reported from -100 to 100
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

    # plot(fraction_data,orc,rev_cum_errors,random_rejection)

    return rejection_ratio


def plot(percentages, orc, errors, random_rejection):
    plt.plot(percentages, orc, lw=2)
    plt.fill_between(percentages, orc, random_rejection, alpha=0.5)
    plt.plot(percentages, errors[::-1], lw=2)
    plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.0)
    plt.plot(percentages, random_rejection, 'k--', lw=2)
    plt.legend(['Oracle', 'Uncertainty', 'Random'])
    plt.xlabel('Percentage of predictions rejected to oracle')
    plt.ylabel('Classification Error (%)')
    plt.show()
    plt.close()

    plt.plot(percentages, orc, lw=2)
    plt.fill_between(percentages, orc, random_rejection, alpha=0.0)
    plt.plot(percentages, errors[::-1], lw=2)
    plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.5)
    plt.plot(percentages, random_rejection, 'k--', lw=2)
    plt.legend(['Oracle', 'Uncertainty', 'Random'])
    plt.xlabel('Percentage of predictions rejected to oracle')
    plt.ylabel('Classification Error (%)')
    plt.show()
    plt.close()