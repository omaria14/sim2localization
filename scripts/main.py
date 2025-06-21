#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import itertools

import rospy
import tf

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
import tf.transformations

from utils.input_output import get_poses_from_file, save_source_scan_and_pose, read_source_scan_and_pose
from utils.conversions_and_operations import sample_poses, pose_msg_from_list, transformation_matrix_from_pose_list
from utils.point_cloud import lineset_from_oriented_bb

from segmentation.segment_ground import segment_ground
from segmentation.segment_dbscan import segment_dbscan

from localization import Localizer

from classification.segment import SourceSegment, TargetSegment
from classification.classifier import RFClassifier

from sim2localization.srv import GetSimulatedScan
from sim2localization.srv import TransformSensor


# CODE SETTINGS
O3D_VIZ = False
GENERATE_SOURCE_SCANS = False
SIMULATED_SCANS_SAMPLES_NUMBER = 2
POSITION_VARIANCE = [4.0, 3.0, 0.1]
YAW_VARIANCE = 0.2618 # radians

# CLASSIFICATION
TRAINING = False
CLF = RFClassifier(model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifier/rf_5.pkl")))
X = []
Y = []
LOAD_TRAIN_DATA=True
TRAIN_SAVE_X_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifier/rf_5_x.txt"))
TRAIN_SAVE_Y_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifier/rf_5_y.txt"))
if TRAINING and LOAD_TRAIN_DATA:
    X = np.loadtxt(TRAIN_SAVE_X_PATH).tolist()  # if you dont want to overwrite, copy the data to the new name :D
    Y = np.loadtxt(TRAIN_SAVE_Y_PATH).tolist()
if not TRAINING:
    CLF.load_model()

# THRESHOLDED ICP
ICP_THRESHOLD = 0.1

# POSES SOURCE SCANS PATH
if not TRAINING:
    GROUND_TRUTH_POSES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ground_truth_poses.txt"))
    SOURCE_SCANS_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/source_scans"))
else:
    GROUND_TRUTH_POSES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ground_truth_poses_training.txt"))
    SOURCE_SCANS_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/source_scans_training"))



# END CODE SETTINGS


if __name__ == "__main__":
    # initialize ros node MAIN and Publishers
    rospy.init_node('MAIN')
    rate = rospy.Rate(0.5)
    source_scan_pub = rospy.Publisher('source_scan', PointCloud2, queue_size=10)
    target_scan_pub = rospy.Publisher('target_scan', PointCloud2, queue_size=10)
    transformed_source_scan_pub = rospy.Publisher('source_scan_transformed', PointCloud2, queue_size=10)
    empty = point_cloud2.create_cloud_xyz32(Header(), [])
    empty.header.frame_id = "velodyne"
    empty.header.stamp = rospy.Time.now()
            

    # initialize the get_simulated_scan service
    # get simulated target scan(s) that represent the ground truth map but as scans, and compare it 
    # with the real "source scan" taken by the robot. 
    rospy.wait_for_service("get_simulated_scan")
    get_sim_scan_srv_proxy = rospy.ServiceProxy("get_simulated_scan", GetSimulatedScan)
    rospy.wait_for_service("transform_sensor")
    transform_sensor_srv_proxy = rospy.ServiceProxy("transform_sensor", TransformSensor)
    

    # read robot trajectory
    # pose in form  x,y,z,(quaternion) x,y,z,w
    ground_truth_poses_path = GROUND_TRUTH_POSES_PATH
    ground_truth_poses = get_poses_from_file(ground_truth_poses_path)

    ##############--------------##############
    # generate source scans
    # As no real data, we use also a simulated scan to generate source_scans. 
    # loop over robot designed poses and apply the algorithm on them
    if GENERATE_SOURCE_SCANS:
        source_scans = []
        # get source_scans from simulation at each pose
        for i in range(len(ground_truth_poses)):
            sensor_pose = Pose()
            sensor_pose.position.x = ground_truth_poses[i][0] 
            sensor_pose.position.y = ground_truth_poses[i][1]
            sensor_pose.position.z = ground_truth_poses[i][2]
            sensor_pose.orientation.x = ground_truth_poses[i][3]
            sensor_pose.orientation.y = ground_truth_poses[i][4]
            sensor_pose.orientation.z = ground_truth_poses[i][5]
            sensor_pose.orientation.w = ground_truth_poses[i][6]
            transform_sensor_srv_proxy(sensor_pose)
            time.sleep(0.5)
            source_scan = get_sim_scan_srv_proxy("test")
            source_scan_save_path = os.path.join(SOURCE_SCANS_SAVE_PATH, f"source_scan_{i}.json")
            save_source_scan_and_pose(source_scan, ground_truth_poses[i], source_scan_save_path)
        exit(0)
    ##############--------------##############

    else:
        ##############--------------##############
        # read source scan and generate sampled target_simulated_scans
        # read pre-generated source scans
        source_scans = []
        source_scans_poses_gt = []
        source_scans_poses_initial_guess = []
        for i in range(len(os.listdir(SOURCE_SCANS_SAVE_PATH))):
            source_scan_path = os.path.join(SOURCE_SCANS_SAVE_PATH, f"source_scan_{i}.json")
            source_scan, pose_gt = read_source_scan_and_pose(source_scan_path)
            source_scans.append(source_scan)
            source_scans_poses_gt.append(pose_gt)
            # we have ground truth from simulation, but corresponinhg transformation is 
            # initially deviated -> generate source_scans_poses_initial_guess
            pose_initial_guess = sample_poses(pose_gt, 1, POSITION_VARIANCE, YAW_VARIANCE)[0]
            source_scans_poses_initial_guess.append(pose_initial_guess)
        
        # for each source scan, generate multiple simulated target scans according to uncertainity
        for i in range(len(source_scans)):

            # get corresponding target simulated scan(s)
            target_poses = sample_poses(source_scans_poses_initial_guess[i], SIMULATED_SCANS_SAMPLES_NUMBER,
                                        POSITION_VARIANCE, YAW_VARIANCE)
            target_simulated_scans = []
            for j in range(len(target_poses)):
                target_pose = target_poses[j]
                target_pose_msg = pose_msg_from_list(target_pose)
                transform_sensor_srv_proxy(target_pose_msg)
                time.sleep(0.5)
                target_simulated_scan = get_sim_scan_srv_proxy("")
                target_simulated_scans.append(target_simulated_scan)
            
            # convert source_scan and target simulated scans from pointcloud2 to o3d
            source_scan_o3d = o3d.geometry.PointCloud()
            source_scan_o3d.points = o3d.utility.Vector3dVector(point_cloud2.read_points_list(source_scans[i]))
            source_scan_trans_initial_guess = transformation_matrix_from_pose_list(source_scans_poses_initial_guess[i])
            source_scan_o3d.transform(source_scan_trans_initial_guess)
            source_scan_o3d.paint_uniform_color([1.0, 0.0, 0])
            target_simulated_scans_o3d = []
            for j in range (len(target_simulated_scans)):
                target_simulated_scan = target_simulated_scans[j]
                target_simulated_scan_o3d = o3d.geometry.PointCloud()
                points_list = point_cloud2.read_points_list(target_simulated_scan.simulated_scan, field_names=["x", "y", "z"])
                points_list = [[point.x, point.y, point.z] for point in points_list]
                target_simulated_scan_o3d.points = o3d.utility.Vector3dVector(points_list)
                target_simulated_scan_trans = transformation_matrix_from_pose_list(target_poses[j])
                target_simulated_scan_o3d.transform(target_simulated_scan_trans)
                target_simulated_scan_o3d.paint_uniform_color([0.0, 1.0, 0.0])
                target_simulated_scans_o3d.append(target_simulated_scan_o3d)
            if O3D_VIZ:
                o3d.visualization.draw_geometries([source_scan_o3d]+target_simulated_scans_o3d, "Source and Target Simulated Scans")

            ##############--------------##############

            ##############--------------##############
            # segmentation and generation of SourceSegments and TargetSegments objects
            # source scan
            plane_model, ground_inliers = segment_ground(copy.deepcopy(source_scan_o3d))
            ground_cloud = copy.deepcopy(source_scan_o3d).select_by_index(ground_inliers)
            outlier_cloud = copy.deepcopy(source_scan_o3d).select_by_index(ground_inliers, invert=True)
            source_segments_o3d = segment_dbscan(copy.deepcopy(outlier_cloud), eps=0.6, min_points=20, min_cluster_points=35)
            if O3D_VIZ:
                o3d.visualization.draw_geometries(source_segments_o3d, "Source Scan Clusters")
            
            # target simulated scans
            target_segments_o3d = []
            for j in range(len(target_simulated_scans_o3d)):
                target_simulated_scan_o3d = copy.deepcopy(target_simulated_scans_o3d[j])
                plane_model, ground_inliers = segment_ground(copy.deepcopy(target_simulated_scan_o3d))
                ground_cloud = copy.deepcopy(target_simulated_scan_o3d).select_by_index(ground_inliers)
                outlier_cloud = copy.deepcopy(target_simulated_scan_o3d).select_by_index(ground_inliers, invert=True)
                target_segments_o3d += segment_dbscan(copy.deepcopy(outlier_cloud), eps=0.6, min_points=20, min_cluster_points=35)
            if O3D_VIZ:
                o3d.visualization.draw_geometries(target_segments_o3d, "Target Simulated Scans Clusters")

            # make the segments in form of objects TargetSegment and SourceSegment
            target_segments = []
            for j in range(len(target_segments_o3d)):
                target_segment = TargetSegment(j, copy.deepcopy(target_segments_o3d[j]), np.array([0,0,0]))
                target_segments.append(target_segment)
            source_segments = []
            for j in range(len(source_segments_o3d)):
                source_segment = SourceSegment(j, copy.deepcopy(source_segments_o3d[j]), np.array([0,0,0]), 10.0, training=TRAINING)
                # get similar candidates of each source segment from target segments
                source_segment.get_similar_segments_candidates(target_segments)
                source_segments.append(source_segment)
            
            # finally get similarity probability for each similar target candidate
            if not TRAINING:
                for j in range(len(source_segments)):
                    source_segments[j].calculate_candidates_similarity_feature_vector()  
                    if len(source_segments[j].similar_candidates)>0:
                        predictions, probabilities = CLF.predict(
                            np.array(source_segments[j].similar_candidates_feature_vectors))
                        source_segments[j].similar_candidates_classification_results = probabilities
            
            # IF TRAINING .. collect data for classifier, don't continue with the rest of localization steps
            if TRAINING:
                for j in range(len(source_segments)):
                    if len(source_segments[j].similar_candidates)>0:
                        x, y = source_segments[j].calculate_candidates_similarity_feature_vector(TRAINING)
                        X += x
                        Y += y     
                        
            ##############--------------##############

            ##############--------------##############
            if not TRAINING:
                # publish target and source scan
                # publish target scan
                target_cloud_rviz_o3d = o3d.geometry.PointCloud()
                for c in target_segments_o3d:
                    target_cloud_rviz_o3d += c
                target_cloud_rviz = point_cloud2.create_cloud_xyz32(Header(), np.asarray(target_cloud_rviz_o3d.points))
                target_cloud_rviz.header.frame_id = "velodyne"
                target_cloud_rviz.header.stamp = rospy.Time.now()
                target_scan_pub.publish(target_cloud_rviz)

                # publish source scan
                # first overwrite transformed 
                transformed_source_scan_pub.publish(empty)
                # source_scan_o3d is already transformed to initial guess
                source_scan_trans_initial_guess = \
                point_cloud2.create_cloud_xyz32(Header(), np.asarray(source_scan_o3d.points))
                source_scan_trans_initial_guess.header.frame_id = "velodyne"
                source_scan_trans_initial_guess.header.stamp = rospy.Time.now()
                source_scan_pub.publish(source_scan_trans_initial_guess)
                ##############--------------##############

                ##############--------------##############
                # calculate global transformation
                # get different possible matches that could suggest a global transformation
                # considering all possible single and pair matches
                single_matches = []
                group_matches = []

                # Iterate over all combinations of 2 source segments
                for source_pair in itertools.combinations(source_segments, 2):
                    # 1) single matches for each source in the pair
                    for source in source_pair:
                        for target in source.similar_candidates:
                            single_matches.append([(source, target)])

                    # 2) group matches (2 source -> 2 target combinations)
                    candidate_lists = [s.similar_candidates for s in source_pair]
                    for t_combo in itertools.product(*candidate_lists):
                        group_matches.append(list(zip(source_pair, t_combo)))
                
                matches = single_matches+group_matches
                
                # global transformation
                best_tranformation, best_score = Localizer.calculate_global_transformation(matches, source_segments)
                source_scan_o3d_transformed = copy.deepcopy(source_scan_o3d)
                source_scan_o3d_transformed.transform(best_tranformation)
                if O3D_VIZ:
                    o3d.visualization.draw_geometries([source_scan_o3d_transformed]+target_simulated_scans_o3d, "Source and Target after global transformation")
                ##############--------------##############
                
                ##############--------------##############
                # apply Thresholded ICP
                icp_result = Localizer.thresholded_icp(source_segments, target_segments, 
                                                    best_tranformation, ICP_THRESHOLD)
                ##############--------------##############
                
                ##############--------------##############
                # publish localization result
                # publish transformed scan # first overwrite source
                source_scan_pub.publish(empty)
                # transform source scan and publish it
                source_scan_o3d_transformed = copy.deepcopy(source_scan_o3d)
                source_scan_o3d_transformed.transform(icp_result.transformation)
                source_scan_full_transform = \
                point_cloud2.create_cloud_xyz32(Header(), np.asarray(source_scan_o3d_transformed.points))
                source_scan_full_transform.header.frame_id = "velodyne"
                source_scan_full_transform.header.stamp = rospy.Time.now()
                transformed_source_scan_pub.publish(source_scan_full_transform)
                ##############--------------##############
        if TRAINING:
            X = np.array(X)
            Y = np.array(Y)
            CLF.train(X,Y)
            CLF.save_model()
            np.savetxt(TRAIN_SAVE_X_PATH, X)
            np.savetxt(TRAIN_SAVE_Y_PATH, Y)
