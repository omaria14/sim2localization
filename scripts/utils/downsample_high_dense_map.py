import open3d as o3d
import os
from point_cloud import range_crop_pcd

if __name__ == "__main__":
    # read pcd
    pcd = o3d.io.read_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/new_map.ply")))
    pcd = range_crop_pcd(pcd, 60, 120, -10, 15, -10, 20)
    pcd = pcd.voxel_down_sample(0.1)
    o3d.io.write_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/new_map_down_test.ply")), pcd)
