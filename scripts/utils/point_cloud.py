#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree

def preprocess_pcd(pcd, crop, crop_center, crop_radius, voxel_size=0.1, max_nn=30):
    preprocessed_pcd = copy.deepcopy(pcd)
    if crop:
        points = np.asarray(preprocessed_pcd.points)               
        kdtree = cKDTree(points)
        indices = kdtree.query_ball_point(crop_center, crop_radius)
        points_crop = points[indices]
        pcd_crop = o3d.geometry.PointCloud()
        pcd_crop.points = o3d.utility.Vector3dVector(points_crop)
        preprocessed_pcd = pcd_crop
    if voxel_size>0.01:
        preprocessed_pcd = preprocessed_pcd.voxel_down_sample(voxel_size)
        radius = voxel_size * 2
    else:
        radius = 0.3     
    preprocessed_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))   
    return preprocessed_pcd

def range_crop_pcd(pcd, x_min, x_max, y_min, y_max, z_min, z_max, o3d_viz=False):
    points = np.asarray(copy.deepcopy(pcd).points)
    ind = []
    for i in range(points.shape[0]):
        if points[i][0] < x_min or points[i][0] > x_max or \
        points[i][1] < y_min or points[i][1] > y_max or \
        points[i][2] < z_min or points[i][2] > z_max:
            ind.append(i)
    pcd = pcd.select_by_index(ind, invert=True)
    if o3d_viz:
        o3d.visualization.draw_geometries([pcd], "pcd range crop")
    return pcd

def lineset_from_oriented_bb(oriented_bb, color=[1.0, 0.0, 0.0]):
    bb_points = np.asarray(oriented_bb.get_box_points())
    lines = [[0, 1], [0, 2], [2, 7], [1, 7],
         [0, 3], [2, 5], [7, 4], [1, 6],
         [3, 5], [3, 6], [5, 4], [4, 6]]

    # Use the same color for all lines
    colors = [color for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bb_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set