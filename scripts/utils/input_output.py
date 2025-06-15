#!/usr/bin/env python3
import os
import numpy as np

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

def get_init_guess_poses(file_path):
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
            pose = np.asarray(pose).reshape((4,4))
            poses.append(pose)
    return poses