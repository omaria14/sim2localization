import open3d as o3d 
import os
import numpy as np
import copy
from point_cloud import range_crop_pcd

if __name__ == "__main__":
    
    # map
    map_full = o3d.io.read_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/new_map_down_test.ply")))
    map_full.paint_uniform_color([0.0, 1.0, 0])
    map_full = map_full.voxel_down_sample(voxel_size=0.05)
    map_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    map_full.orient_normals_consistent_tangent_plane(30)
    o3d.visualization.draw_geometries([map_full], zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(map_full, 0.4)
    o3d.visualization.draw_geometries([mesh], "MESH")
    o3d.io.write_triangle_mesh(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test.ply")), mesh)
