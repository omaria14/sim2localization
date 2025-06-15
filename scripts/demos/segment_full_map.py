import numpy as np
import open3d as o3d
import copy
from segmentation.segment_dbscan import segment_dbscan

def segment_full_map(full_map, ground_plane_model):
    # remove ground and ceiling
    points = np.asarray(copy.deepcopy(full_map).points)
    ind = []
    for i in range(points.shape[0]):
        dist_with_ground = abs(full_map.points[i][0]*ground_plane_model[0] +
            full_map.points[i][1]*ground_plane_model[1] +
            full_map.points[i][2]*ground_plane_model[2] +
            ground_plane_model[3])
        dist_with_ceiling = abs(full_map.points[i][0]*ground_plane_model[0] +
            full_map.points[i][1]*ground_plane_model[1] +
            full_map.points[i][2]*ground_plane_model[2] +
            (ground_plane_model[3] - 9.0))
        if  dist_with_ground < 0.3 or dist_with_ceiling < 3.5:
            ind.append(i)
    full_map = full_map.select_by_index(ind, invert=True)
    o3d.visualization.draw_geometries([full_map], "full map after removing ground and ceiling")
    # db scan
    clusters_clouds = segment_dbscan(full_map)
    o3d.visualization.draw_geometries(clusters_clouds, "full map clusters")
    return clusters_clouds
    