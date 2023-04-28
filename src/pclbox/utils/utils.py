#!/usr/bin/env python3

# -----------------------
#
# File containing utility functions for point cloud processing 
# 
# Author: Wouter Van den Broeck (wouter.vandenbroeck@ugent.be)
# Created March 2023
#
# -----------------------

import logging
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

__all__ = ['read_clouds', 'combine_pcds', 'get_bbox', 'cloud2voxel', 'cloud2mesh']

# Set logger
logger = logging.getLogger(__name__)


def read_clouds(files_in, n=None, random=False, down_sample_size=None, color=False):
    """
    Read in point cloud files 
    
    Args:
        files_in (string or list): 
            if string: directory containing point clouds
            if list: list with filenames of point clouds
        n (int): 
            integer specicfying number of files to read. If n=None all files are read.
        random (bool): 
            if true n random files are taken, else the first n files are taken
        down_sample_size (float): 
            if specified, the point cloud will be voxel-down sampled using the given value 
        color (bool): 
            if true, color each indivual point cloud with random color

    Returns:
        (list(open3d.geometry.PointCloud)): list of point clouds
    """
    # allowed file formats
    FORMATS = ['ply']

    if isinstance(files_in, str):
        # Assume string to be directory path
        if os.path.isdir(files_in):
            # Collect all files in directory of allowed FORMAT in list
            files_in = [files_in + f for f in os.listdir(files_in) if f[-3:] in FORMATS]
        else:
            raise ValueError('Provided directory path does not exist.')
    elif isinstance(files_in, list):
        # Assume files_in to be list of filepaths
        if not all(os.path.isfile(file) for file in files_in):
            raise ValueError('Provided paths are not all valid.')
    else:
        raise ValueError('Files_in must be a directory or a list containing filenames.')
    
    # Get n files
    if n is not None:
        files_in = files_in[:n] if random == False else np.random.choice(files_in, size=n, replace=False)

    # Read in point clouds of n files and give random color
    clouds = []
    for i in tqdm(range(len(files_in))):
        pcl = o3d.io.read_point_cloud(files_in[i])

        # Optional down sampling
        if down_sample_size is not None:
            pcl = pcl.voxel_down_sample(down_sample_size)

        if color:
            col = np.random.randint(0, 256, 3) / 255
            pcl.paint_uniform_color(col)

        clouds.append(pcl)

    return clouds


def combine_pcds(files_in, path_out=False):
    """ 
    Combine point clouds to one single point cloud 
    """

    # If files_in is list of in memory point clouds, move on
    if isinstance(files_in, list) and all(isinstance(file, o3d.geometry.PointCloud) for file in files_in):
        pcds = files_in
    else:
        # Read in all point clouds in given directory or list with paths to pcds
        logger.info('Reading point clouds...')
        pcds = read_clouds(files_in)
    
    # Combine point clouds to single pcl
    pcd_combined = o3d.geometry.PointCloud()
    logger.info('combining point clouds...')
    for i in tqdm(range(len(pcds))):
        pcd_combined += pcds[i]

    # Return combined point cloud or write to disk
    if not path_out:
        return pcd_combined
    else:
        logger.info('Writing point cloud...')
        o3d.io.write_point_cloud(
            path_out,
            pcd_combined,
            write_ascii=True,
            compressed=False,
            print_progress=True,
        )


def get_bbox(pcl, compute_overall_bbox=False, path_out=None):
    """
    Get bounding box(es) of point cloud(s) and optionally write to csv file 

    Args:
        pcl (open3d.geometry.PointCloud or list(open3d.geometry.PointCloud)): (list of) point cloud(s)
        compute_overall_bbox (bool): if true, compute the single overall bounding box of all point clouds
        path_out: if specified, write result to csv file

    Returns:
        (pandas.DataFrame): dataframe with bounding box coordinates of point cloud(s)

    """
    if isinstance(pcl, o3d.geometry.PointCloud):
        pcl = [pcl]

    bbox_min = np.array([np.min(np.asarray(p.points), axis=0) for p in pcl])
    bbox_max = np.array([np.max(np.asarray(p.points), axis=0) for p in pcl])

    if compute_overall_bbox:
        bbox_min = np.min(bbox_min, axis=0, keepdims=True)
        bbox_max = np.max(bbox_max, axis=0, keepdims=True)

    bbox = np.hstack((bbox_min, bbox_max))

    # Convert to dataframe and save to csv file
    columns = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max'] 
    bbox_df = pd.DataFrame(data=bbox, columns=columns)

    if path_out is not None:
        bbox_df.to_csv(path_out, index=False)
    
    return bbox_df
        

def cloud2voxel(trees, voxel_size=0.1):
    """
    Convert all point clouds to voxel representation
    """
    return [o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size=voxel_size) for pcl in trees]


def cloud2mesh(pcl, algorithm='ball_pivoting'):
    """
    Convert point cloud to mesh    
    """
    # Input arg checks
    if algorithm not in ['ball_pivoting', 'poisson']:
        raise ValueError("'algorithm' must be one of 'ball_pivoting' or 'poisson'")

    if algorithm == 'ball_pivoting':
        # Estimate good ball radius 
        distances = pcl.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        # Run ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcl,
            o3d.utility.DoubleVector([radius, radius * 2])
        )

        # Remove artifacts
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    elif algorithm == 'poisson':
        # Surface reconstruction Poisson algorithm
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=False,
        )

        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Crop to clean artifacts
        bbox = pcl.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    return mesh