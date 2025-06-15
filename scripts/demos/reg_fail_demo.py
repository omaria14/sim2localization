#!/usr/bin/env python3

"""
A script for showing a demo how the registering point cloud scan into a map could fail if the map changes. 
"""
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

START_SCAN = 850
SCAN_SAMPLING = 20
PROCESS_SCANS_NO = 200
O3D_VIS = True


if __name__ == "__main__":
    # initialize ros node, publishers
    rospy.init_node('reg_fail_demo')
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
    # for each scan ..
    for i in range(PROCESS_SCANS_NO):
        # rewrite map
        map = copy.deepcopy(map_full)
        # retrieve the scan
        scan_path = scan_paths[START_SCAN+i*SCAN_SAMPLING]
        scan_points, _ = read_scan(scan_path)
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(scan_points)
        # get initial pose
        initial_guess = init_guess_poses[START_SCAN+(i*SCAN_SAMPLING)-SCAN_SAMPLING]
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
        scan_initial_guess = copy.deepcopy(scan)
        scan_initial_guess.transform(initial_guess)
        scan_registered = copy.deepcopy(scan)
        scan_registered.transform(reg_p2l.transformation)
        # o3d visualize result
        if O3D_VIS:
            pose_truth = init_guess_poses[START_SCAN+i*SCAN_SAMPLING]  # truth? -> 
            # registered with high sampling rate and compared with SLAM trajectory (positions from SLAM trajectory) 
            scan_registered_truth = copy.deepcopy(scan)
            scan_registered_truth.transform(pose_truth)
            scan_initial_guess.paint_uniform_color([0.0, 1.0, 0])
            scan_registered.paint_uniform_color([1.0, 0.0, 0])
            scan_registered_truth.paint_uniform_color([0.0, 0.0, 1.0])
            map.paint_uniform_color([0.0, 1.0, 0])
            scan_initial_guess.paint_uniform_color([0.0, 0.0, 0.0])
            # o3d.visualization.draw_geometries([scan_initial_guess], "scan")
            o3d.visualization.draw_geometries([map, scan_initial_guess], "cropped map")
            # o3d.visualization.draw_geometries([scan_initial_guess, scan_registered_truth])
            # o3d.visualization.draw_geometries([scan_registered, scan_registered_truth])
        # publish result
        map_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(map_full.points))
        scan_init_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(scan_initial_guess.points))
        scan_registered_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(scan_registered.points))
        map_pub.publish(map_ros)
        scan_init_pub.publish(scan_init_ros)
        scan_registered_pub.publish(scan_registered_ros)