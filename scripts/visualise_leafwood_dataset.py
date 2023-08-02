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
    2: [0, 0, 1],
    3: [0, 0, 0],
}

def load_dataset(dataset_path):

    pcd_names = glob.glob(dataset_path + '/rc*' )

    pcds = []
    for i in tqdm(range(len(pcd_names)), desc='loading dataset'):
        cloud = np.loadtxt(os.path.join(dataset_path, pcd_names[i]))
        points = cloud[:, :3]
        labels = cloud[:, 3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # pcd = pcd.voxel_down_sample(0.1)
        # col = np.random.randint(0, 256, 3) / 255
        # pcd.paint_uniform_color(col)

        colors = [label2color[label] for label in labels]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcds.append(pcd)
    
    return pcds


if __name__ == '__main__':

    # dataset = load_dataset(DATASET_PATH)
    # o3d.visualization.draw_geometries(dataset)

    # pcl_path = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/leaf_wood/preprocessed/oc_011_pc.txt"
    # pcl_path_pred = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/leaf_wood/preprocessed_open3d/prediction/oc_011_pc.txt"
    pcl_path = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/leafon_individual_TLSdata/prediction/303_T0_2_175c.txt"
    
    cloud = np.loadtxt(pcl_path)
    # cloud_pred = np.loadtxt(pcl_path_pred)

    points = cloud[:, :3]
    labels = cloud[:, 3]

    # points_pred = cloud_pred[:, :3]
    # labels_pred = cloud_pred[:, 3]

    # label_eval = np.zeros((labels.shape[0], 1), dtype=np.uint8)
    # label_eval[((labels == 0) & (labels_pred == 0))] = 0
    # label_eval[((labels == 1) & (labels_pred == 1))] = 1
    # label_eval[((labels == 1) & (labels_pred == 0))] = 2
    # label_eval[((labels == 0) & (labels_pred == 1))] = 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # colors = [label2color[label] for label in label_eval.flatten()]
    colors = [label2color[label] for label in labels]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


