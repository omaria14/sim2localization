import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt


def segment_dbscan(input_cloud, eps=0.5, min_points=10, min_cluster_points=40):
    # WIP: DBSCAN
    # segment outlier cloud using DBSCAN algorithm
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(copy.deepcopy(input_cloud).cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    input_points = np.asarray(copy.deepcopy(input_cloud.points))
    clusters_points = []
    clusters_clouds = []
    for j in range(labels.max()+1):
        cluster_points = copy.deepcopy(input_points)[labels == j]
        if cluster_points.shape[0] > min_cluster_points:
            clusters_points.append(cluster_points)
    cmap = plt.cm.get_cmap("tab20", len(clusters_points))
    for j in range(len(clusters_points)):
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(clusters_points[j])
        color = cmap(j)
        cluster_cloud.paint_uniform_color(color[:3])
        clusters_clouds.append(cluster_cloud)
    return clusters_clouds
