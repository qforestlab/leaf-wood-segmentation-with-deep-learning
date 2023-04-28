#!/usr/bin/env python3

import os
import random
import numpy as np
import pandas as pd
import open3d as o3d
from pclbox.utils.utils import read_clouds, cloud2voxel, cloud2mesh

def visualize():
    datafolder = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/segmented/"
    n_trees = None
    down_sample_size = 0.1
    trees = read_clouds(datafolder, n=n_trees, down_sample_size=down_sample_size, color=True)
    o3d.visualization.draw_geometries(trees)

if __name__ == '__main__':
    
    visualize()

    # datafolder_segm = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/segmented/"

    # Specify path to datafolder and store all filenames in list
    # datafolder = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/data/"
    
    # tile_names = [
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_25.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_26.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_27.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_28.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_29.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_30.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_44.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_158.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_159.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_160.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_161.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_162.ply",
    #     "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/wytham_winter_163.ply",
    # ]
    
    # down_sample_size = 0.1
    # tiles = [o3d.io.read_point_cloud(tile).voxel_down_sample(voxel_size=down_sample_size) for tile in tile_names]

    # datafolder_segm = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/segmented/"
    # filenames = [f for f in os.listdir(datafolder_segm) if f[-3:] == 'ply']
    # # filenames = filenames[::2]

    # trees = []
    # for filename in filenames:
    #     pcl = o3d.io.read_point_cloud(datafolder_segm + filename)
    #     color = np.random.randint(0, 256, 3) / 255
    #     pcl.paint_uniform_color(color)

    #     pcl = pcl.voxel_down_sample(voxel_size=down_sample_size)

    #     trees.append(pcl)

    # scene = tiles + trees
    # o3d.visualization.draw_geometries(scene)

    # # Compute number of trees
    # datafolder_segm = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/segmented/"
    # nr_trees = len([f for f in os.listdir(datafolder_segm) if f[-3:] == 'ply'])

    # # Make colormap, mapping tree number to a unique color, -1 = non-tree
    # colors = np.random.randint(0, 256, (nr_trees, 3)) / 255
    # colormap = {i: color for i, color in enumerate(colors)}
    # colormap[-1] = np.array([0., 0., 0.])

    # # Read in all tiles
    # datafolder_raw_clipped = "/mnt/c/Users/wavdnbro/OneDrive - UGent/Documents/spacetwin/datasets/wythamwoods/raw_selection_clipped_10-25/"
    # tilenames = [f for f in os.listdir(datafolder_raw_clipped) if f[-3:] == 'ply']
    # # tiles = [o3d.io.read_point_cloud(datafolder_raw_clipped + tile) for tile in tilenames]

    # # Loop over all tiles
    # tiles = []
    # n_tiles = 30
    # for tilename in tilenames[:n_tiles]:
    #     # Read tile
    #     tile = o3d.io.read_point_cloud(datafolder_raw_clipped + tilename)

    #     # Read in file with labels for all points in tile
    #     labels = pd.read_csv(datafolder_raw_clipped + tilename[:-4] + '_label.csv')

    #     # Map point labels to color
    #     colors = [colormap[label] for label in labels['label_instance']]
    #     tile.colors = o3d.utility.Vector3dVector(colors)

    #     # Optional downsampling
    #     tile = tile.voxel_down_sample(0.05)    

    #     tiles.append(tile)

    # o3d.visualization.draw_geometries(tiles)

    # Read in point clouds
    # n = 2
    # trees = read_clouds(datafolder, n)

    # Convert to voxel representation
    # voxel_size = 0.1
    # trees_voxels = cloud2voxel(trees, voxel_size)
    # o3d.visualization.draw_geometries(trees_voxels)


    # Estimate normals for first tree and make consistent
    # trees[0].estimate_normals()
    # trees[0].orient_normals_consistent_tangent_plane(100)

    # Visualize point clouds
    # o3d.visualization.draw_geometries([trees[0]], point_show_normal=True)
    # o3d.visualization.draw_geometries(trees)

    # Surface reconstruction from point cloud
    # mesh = cloud2mesh(trees[0], algorithm='poisson')

    # Visualize point clouds
    # o3d.visualization.draw_geometries([mesh])

