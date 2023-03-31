#!/usr/bin/env python3

# ----------------------------------------------------
#
# Script to visualise the tropical leaf - wood dataset
#
# ----------------------------------------------------

import os
import glob
import numpy as np
from tqdm import tqdm
import open3d as o3d

DATASET_PATH = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/leaf_wood/preprocessed/"

label2color = {
    0: [0, 1, 0],
    1: [1, 0, 0],
}

def load_dataset(dataset_path):

    pcd_names = glob.glob(dataset_path + '/dro*' )

    pcds = []
    for i in tqdm(range(len(pcd_names)), desc='loading dataset'):
        cloud = np.loadtxt(os.path.join(dataset_path, pcd_names[i]))
        points = cloud[:, :3]
        labels = cloud[:, 3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        colors = [label2color[label] for label in labels]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcds.append(pcd)
    
    return pcds


if __name__ == '__main__':

    dataset = load_dataset(DATASET_PATH)
    o3d.visualization.draw_geometries(dataset)