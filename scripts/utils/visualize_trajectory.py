import os
import numpy as np
import open3d as o3d

if __name__=="__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/conslam_seq4/trajectory_gt_poses_normalized"))
    files = os.listdir(path=parent_dir)
    def extract_numeric(file_name):
        base_name, ext = os.path.splitext(file_name)
        return int(base_name) if base_name.isdigit() else base_name
    sorted_files = sorted(files, key=extract_numeric)
    frames = o3d.geometry.TriangleMesh()
    for i in range (len(sorted_files)):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        transform = np.loadtxt(os.path.join(parent_dir, sorted_files[i]))
        frame.transform(transform)
        frames += frame
        if i%50 == 0:
            o3d.visualization.draw_geometries([frames], "Trajectory")        
    o3d.visualization.draw_geometries([frames], "Trajectory")
   # o3d.io.write_triangle_mesh(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/conslam_seq3/trajectory_vis.ply")), frames)

    