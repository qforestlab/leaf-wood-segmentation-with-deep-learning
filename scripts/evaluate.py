#!/usr/bin/env python3

# ----------------------------------------------------
#
# Script to evaluate the predictions 
#
# Expected file organization:
#   dataset_folder
#       |- test
#       |   |- cloud1_labels.txt
#       |   |- cloud2_labels.txt
#       |   |- ...
#       |- prediction
#       |   |- cloud1.txt
#       |   |- cloud2.txt
#       |   |- ...
#
# Usage: type the following in a linux terminal: python3 scripts/evaluate.py cfg/<config_file_infer.yml> 
#
#
# ----------------------------------------------------

import os
import numpy as np
import argparse
import ml3d as _ml3d
import ml3d.torch as ml3d


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help='path to configuration file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # Load config file
    args = get_arguments()
    if args.cfg is not None:
        cfg_path = args.cfg
        cfg = _ml3d.utils.Config.load_from_file(cfg_path)

    # Get data path and test folder from config file
    data_path = cfg.dataset.get('dataset_path')
    test_dir = cfg.dataset.get('test_dir')
    test_pred_dir = cfg.dataset.get('test_result_folder', 'test_pred')
    path_true = os.path.join(data_path, test_dir)
    path_pred = os.path.join(data_path, test_pred_dir)

    pred_filenames = os.listdir(path_pred)
    true_filenames = [filename[:-4] + '_labels.txt' for filename in pred_filenames]

    # Loop over ground truth and predictions and keep track of metrics
    TP, FP, TN, FN = 0, 0, 0, 0

    for gt_file, pred_file in zip(true_filenames, pred_filenames):
        # Load ground truth and prediction
        gt = np.loadtxt(os.path.join(path_true, gt_file))
        pred = np.loadtxt(os.path.join(path_pred, pred_file))

        # get label
        gt = gt.flatten()
        pred = pred[:, 3].flatten()

        TP += ((gt == 1) & (pred == 1)).sum()
        TN += ((gt == 0) & (pred == 0)).sum()
        FP += ((gt == 0) & (pred == 1)).sum()
        FN += ((gt == 1) & (pred == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    f1 = (2 * TP) / (2 * TP + FP + FN)

    print(
        'TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, '\n'
        'accuracy:', accuracy, '\n',
        'precision:', precision, '\n',
        'recall:', recall, '\n',
        'iou:', iou, '\n',
        'f1:', f1, '\n',
    )


