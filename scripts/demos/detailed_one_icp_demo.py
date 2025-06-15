#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import copy

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from utils.input_output import get_scan_paths, read_scan, get_init_guess_poses
from utils.point_cloud import preprocess_pcd

SCAN_TO_PROCESS = 850
INITIAL_GUESS_OFFSET = 20


if __name__ == "__main__":
    # initialize ros node, publishers
    rospy.init_node('detailed_one_icp_demo')
    map_pub = rospy.Publisher('map', PointCloud2, queue_size=10)
    scan_init_pub = rospy.Publisher('scan_init', PointCloud2, queue_size=10)
    scan_registered_pub = rospy.Publisher('scan_registered', PointCloud2, queue_size=10)
    # read required data
    # map
    map_full = o3d.io.read_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/map.pcd")))
    # o3d.visualization.draw_geometries([map])
    # scans for a trajectory within the map after update -> to be registered
    scan_paths = \
    get_scan_paths(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/lidar_scans")))
    # initial guess poses
    # poses that could be used for initial guess. Those poses are retrieved from a correct registration of 
    # the scans into the map, so if used for initial guess -> use previous poses with required indexing.
    # e.g. indexing 5 & start scan 100 -> use scans 90, 95 to estimate the initial guess (linear velocity model).
    init_guess_poses = \
    get_init_guess_poses(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/poses_for_init_guess.txt")))
    
    # start registration sequence
    

    # rewrite map
    map = copy.deepcopy(map_full)
    # retrieve the scan
    scan_path = scan_paths[SCAN_TO_PROCESS]
    scan_points, _ = read_scan(scan_path)
    scan = o3d.geometry.PointCloud()
    scan.points = o3d.utility.Vector3dVector(scan_points)
    # get initial pose
    initial_guess = init_guess_poses[SCAN_TO_PROCESS-INITIAL_GUESS_OFFSET]
    # preprocess map and scan
    map = preprocess_pcd(map, True, initial_guess[:3,3], 35)
    scan = preprocess_pcd(scan, True, np.array([0.0,0.0,0.0]), 15)
    # register the frame
    sigma = 2.0
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma / 3)
    max_correspondance_distance=2 * sigma
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        scan, map, max_correspondance_distance, initial_guess, p2l) 
    #### correspondencese = [...,[source_index, tearget_index], ....] # source = frame, target = map
    scan_initial_guess = copy.deepcopy(scan).transform(initial_guess)
    correspondence_set = np.asarray(reg_p2l.correspondence_set)
    correspondence_set_points = []
    correspondence_set_lines = []
    frame_points = np.asarray(map.points)
    source_points = np.asarray(scan_initial_guess.points)
    for j in range(correspondence_set.shape[0]-1):
        correspondence_set_points.append(source_points[correspondence_set[j][0]])
        correspondence_set_points.append(frame_points[correspondence_set[j][1]])
        correspondence_set_lines.append([len(correspondence_set_points)-1, len(correspondence_set_points)-2])
    colors = [[1, 0, 0] for i in range(len(correspondence_set_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(correspondence_set_points),
        lines=o3d.utility.Vector2iVector(correspondence_set_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # visualize the results
    scan_initial_guess.paint_uniform_color([0.0, 0.0, 0])
    map.paint_uniform_color([0.0, 1.0, 0])
    o3d.visualization.draw_geometries([line_set, scan_initial_guess, map])
    o3d.visualization.draw_geometries([line_set])