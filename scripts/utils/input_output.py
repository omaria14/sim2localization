#!/usr/bin/env python3
import os
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
import json

def get_scan_paths(parent_path):
    """
    A function that takes the path of scans folder and returns list of full paths of the scans files
    sorted using the numiric part.
    """
    # get full paths of the scans and sort them
    scan_files = [filename for filename in os.listdir(parent_path) if filename.endswith(".csv")]
    numeric_part = lambda filename: int(''.join(filter(str.isdigit, filename)))
    scan_files = sorted(scan_files, key=numeric_part)
    scan_paths = []
    for i in range(len(scan_files)):
        scan_paths.append(os.path.join(parent_path, scan_files[i]))
    return scan_paths

def read_scan(scan_path):
    """
    """
    with open(scan_path, 'r') as file:
        lines = file.readlines()[1:]
    scan = []
    timestamps = []
    for i in range(len(lines)):
        values = lines[i].strip().split(',')
        timestamps.append(float(values[5]))
        scan.append([float(value) for value in values[7:10]])
    timestamps = np.array(timestamps)
    scan = np.asarray(scan)
    return scan, timestamps

def get_poses_from_file(file_path):
    """
    """
    poses = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line_elements = lines[i].split(",")
            line_elements = line_elements[:-1]  # remove last element as it is a \n character
            pose = []
            for j in range (len(line_elements)):
                pose.append(float(line_elements[j]))
            poses.append(pose)
    return poses

def save_source_scan_and_pose(source_scan, pose, file_path):
    """
    saves PointCloud2 and Pose
    """
    with open(file_path, 'w') as f:
        points_list = point_cloud2.read_points_list(source_scan.simulated_scan, field_names=["x", "y", "z"])
        points_list = [[point.x, point.y, point.z] for point in points_list]
        json.dump({"points": points_list, "pose": pose}, f)

def read_source_scan_and_pose(file_path, point_cloud_type="PointCloud2"):
    """
    return PointCloud2 or o3d.geometry.PointCloud() and Pose
    """
    with open(file_path, 'r') as f:
        data = json.load( f)
    points_list = data["points"]
    fields = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    header = Header()
    source_scan = point_cloud2.create_cloud(header, fields, points_list)
    pose = data["pose"]
    return source_scan, pose

