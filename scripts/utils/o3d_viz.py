import open3d as o3d
import os
import numpy as np






if __name__ == "__main__":
    # pcd = o3d.io.read_triangle_mesh(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/conslam_seq3/sonseq3_reconstructed_map_mesh.ply")))
    # o3d.visualization.draw_geometries([pcd], "Trajectory")


    #pcd = o3d.io.read_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/new_map.ply")))
    #o3d.visualization.draw_geometries([pcd], "Map Point Cloud")
    adj = np.eye(4)
    adj[:3, 3] = [3.0, 0, 0]
    print(np.linalg.inv(adj))
