#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import rospy
import tf

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

import tf.transformations
from utils.input_output import get_scan_paths, read_scan, get_init_guess_poses
from utils.point_cloud import preprocess_pcd, lineset_from_oriented_bb
from segmentation.segment_ground import segment_ground
from segmentation.segment_dbscan import segment_dbscan
from weighted_icp.scripts.classification.segment import SourceSegment, TargetSegment
from classification.classifier import RFClassifier
from demos.segment_full_map import segment_full_map
from icp import WeightedICP

from weighted_icp.srv import GetSimulatedScan
from weighted_icp.srv import TransformSensor


# CODE SETTINGS

# INPUT SCANS
START_SCAN = 670  # 690:860 reasonable
SCAN_SAMPLING = 5
PROCESS_SCANS_NO = 20
O3D_VIS = ["final_result"] # "final_result", "eval_vanilla_icp"
INITIAL_GUESS_ERROR_OFFSET = 30
SCAN2_OFFSET = 10

# TRAIN/INFERENCE
TRAINING=True
LOAD_TRAIN_DATA=False
TRAIN_SAVE_X_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifiers/rf_3_x.txt"))
TRAIN_SAVE_Y_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifiers/rf_3_y.txt"))
CLF = RFClassifier(model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/classifiers/rf_4.pkl")))
if not TRAINING:
    CLF.load_model()
X = []
Y = []
if TRAINING and LOAD_TRAIN_DATA:
    X = np.loadtxt(TRAIN_SAVE_X_PATH).tolist()  # if you dont want to overwrite, copy the data to the new name :D
    Y = np.loadtxt(TRAIN_SAVE_Y_PATH).tolist()

# Evaluation
EVALUATION = False
ground_truth_all_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ground-truth.txt"))
ground_truth_all_matrix = np.genfromtxt(ground_truth_all_path, delimiter=",").reshape(-1, 4, 4)
GROUND_TRUTH = []
ESTIMATION_OURS = []
ESTIMATION_ICP = []
INIT_GUESS_DEVIATION = []
GROUND_TRUTH_SAVE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/evaluation/gt_690_1_100_40.txt"))
ESTIMATION_OURS_SAVE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/evaluation/eo_690_1_100_40.txt"))
ESTIMATION_ICP_SAVE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/evaluation/ei_690_1_100_40.txt"))
INIT_GUESS_DEVIATION_SAVE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/evaluation/id_690_1_100_40.txt"))

if __name__ == "__main__":
    # initialize ros node, publishers
    rospy.init_node('weighte_icp_main')
    map_pub = rospy.Publisher('map', PointCloud2, queue_size=10)
    scan_init_pub = rospy.Publisher('scan_init', PointCloud2, queue_size=10)
    scan_registered_pub = rospy.Publisher('scan_registered', PointCloud2, queue_size=10)
    # read required data
    # scans for a trajectory within the map after update -> to be registered
    scan_paths = \
    get_scan_paths(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/lidar_scans")))
    # initial guess poses
    # poses that could be used for initial guess. Those poses are retrieved from a correct registration of 
    # the scans into the map, so if used for initial guess -> use previous poses with required indexing.
    # e.g. indexing 5 & start scan 100 -> use scans 90, 95 to estimate the initial guess (linear velocity model).
    init_guess_poses = \
    get_init_guess_poses(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/poses_for_init_guess.txt")))
    
    # start registration sequence
    # for each scan ..
    for i in range(PROCESS_SCANS_NO):
        # get initial pose
        initial_guess = init_guess_poses[START_SCAN+(i*SCAN_SAMPLING)-INITIAL_GUESS_ERROR_OFFSET]
        GROUND_TRUTH.append(ground_truth_all_matrix[START_SCAN+(i*SCAN_SAMPLING)])
        INIT_GUESS_DEVIATION.append(initial_guess)        
        ##############
        # retrieve the Current SCAN .... SOURCE SCAN TO BE REGISTERED
        print("SCAN NUMBER:", START_SCAN+i*SCAN_SAMPLING)
        scan_path = scan_paths[START_SCAN+i*SCAN_SAMPLING]
        scan_points, _ = read_scan(scan_path)
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(scan_points)
        # preprocess scan
        scan = preprocess_pcd(copy.deepcopy(scan), True, np.array([0.0,0.0,0.0]), 20)
        # scan.transform(initial_guess)
        scan.paint_uniform_color([1.0, 0.0, 0])
        if "source_scan" in O3D_VIS:
            o3d.visualization.draw_geometries([scan], "Source Scan")
        # segment the scan
        # ground segmentation
        plane_model, ground_inliers = segment_ground(copy.deepcopy(scan), 
                                                     o3d_vis=True if "source_ground_details" in O3D_VIS else False)
        ground_cloud = copy.deepcopy(scan).select_by_index(ground_inliers)
        outlier_cloud = copy.deepcopy(scan).select_by_index(ground_inliers, invert=True)
        ground_cloud.paint_uniform_color([0.0, 0.0, 0])
        outlier_cloud.paint_uniform_color([0.0, 1.0, 0])
        if "source_ground" in O3D_VIS:
            o3d.visualization.draw_geometries([ground_cloud, outlier_cloud], "segmented plane with outlier cloud")
        # DBSCAN
        # segment outlier cloud using DBSCAN algorithm
        source_clusters_clouds = segment_dbscan(copy.deepcopy(outlier_cloud), eps=0.6, min_points=20, min_cluster_points=35)
        if "source_dbscan" in O3D_VIS:
            o3d.visualization.draw_geometries(source_clusters_clouds, "source clusters")
        
        # TARGET SCAN ... Could be SIMULATED SCAN (Localization Test) or other scan with offset to register into (Scan Alignment Test) 
        # SIMULATED SCAN 
        # Set sensor pose through: gazebo/set_model_state
        rospy.wait_for_service("transform_sensor")
        transform_sensor_srv_proxy = rospy.ServiceProxy("transform_sensor", TransformSensor)
        # temp adjustment for multiple simulated scans
        sim_poses = [initial_guess]  # poses for sensor .. initial guess + addidtional poses
        adj = np.eye(4)
        adj[:3, 3] = [3.0, 0, 0]  # aditional pose 1 -> +translation
        sim_poses.append(adj@initial_guess)
        sim_transform_back = [np.eye(4), np.linalg.inv(initial_guess)@adj@initial_guess]  
        # sim_transform_back := transformation matrices to align additial simulation poses with
        # the simulated scan at the initial guess, represented in initial guess local frame.
        # thus -> go to the additional frame where the local additional simulated scan is there .. 
        # return it to be local of initial guess np.linalg.inv(initial_guess)

        # aditional pose 2 -> -translation
        adj[:3,3] = [-3, 0, 0] 
        sim_poses.append(adj@initial_guess)
        sim_transform_back.append(np.linalg.inv(initial_guess)@adj@initial_guess)
        list_target_clusters_clouds = []
        for adj_i, sim_pose in enumerate(sim_poses):
            rot_matrix = np.eye(4)
            rot_matrix[:3, :3] = copy.deepcopy(sim_pose[0:3, 0:3])
            sensor_orientation = tf.transformations.quaternion_from_matrix(rot_matrix)
            sensor_pose = Pose()
            sensor_pose.position.x = sim_pose[0,3] 
            sensor_pose.position.y = sim_pose[1,3]
            sensor_pose.position.z = sim_pose[2,3]
            sensor_pose.orientation.x = sensor_orientation[0]
            sensor_pose.orientation.y = sensor_orientation[1]
            sensor_pose.orientation.z = sensor_orientation[2]
            sensor_pose.orientation.w = sensor_orientation[3]
            transform_sensor_srv_proxy(sensor_pose)
            time.sleep(1.0)
            # Get Simulated Scan. 
            # Note: Simulated Scan is wrt the sensor (after transforming it to initial guess position),
            # so we have to transform also the resulting cloud with the initial guess .. HERE:
            # the resulting points after the transformation are the map points (OUR REFERENCE, which we want
            # to resgister the scan to it) (no matter if the initial guess is not accurate we are seeing 
            # the map also, different initial guesses would only affect how much we see from the map and
            # how the segments look like not the points positions).
            rospy.wait_for_service("get_simulated_scan")
            get_sim_scan_srv_proxy = rospy.ServiceProxy("get_simulated_scan", GetSimulatedScan)
            simulated_scan_pcd2 = get_sim_scan_srv_proxy("test")
            # .. transform to o3d ..
            points_list = point_cloud2.read_points_list(simulated_scan_pcd2.simulated_scan, field_names=["x", "y", "z"])
            points_list = [[point.x, point.y, point.z] for point in points_list]
            simulated_scan = o3d.geometry.PointCloud()
            simulated_scan.points = o3d.utility.Vector3dVector(points_list)
            simulated_scan = preprocess_pcd(copy.deepcopy(simulated_scan), True, np.array([0.0,0.0,0.0]), 20)
            simulated_scan.paint_uniform_color([0.0, 1.0, 0])
            simulated_scan.transform(sim_transform_back[adj_i])
            if "simulated_scan" in O3D_VIS:
                o3d.visualization.draw_geometries([simulated_scan, scan], f"Simulated Scan at Position: {initial_guess[:3, 3]}")
            # .. segment .. 
            plane_model, ground_inliers = segment_ground(copy.deepcopy(simulated_scan), distance_threshold=0.3, 
                                                        o3d_vis=True if "target__ground_details" in O3D_VIS else False)
            simulated_scan_ground_cloud = copy.deepcopy(simulated_scan).select_by_index(ground_inliers)
            outlier_cloud = copy.deepcopy(simulated_scan).select_by_index(ground_inliers, invert=True)
            simulated_scan_ground_cloud.paint_uniform_color([0.0, 0.0, 0])
            outlier_cloud.paint_uniform_color([0.0, 1.0, 0])
            if "target_ground" in O3D_VIS:
                o3d.visualization.draw_geometries([simulated_scan_ground_cloud], "simulated scan segmented plane")
            if "target_ground_with_outlier" in O3D_VIS:
                o3d.visualization.draw_geometries([simulated_scan_ground_cloud, outlier_cloud], "simulated scan segmented plane with outlier cloud")
            # segment outlier cloud using DBSCAN algorithm
            target_clusters_clouds = segment_dbscan(copy.deepcopy(outlier_cloud), eps=0.6, min_points=15, min_cluster_points=35)
            for cluster in target_clusters_clouds:
                cluster.paint_uniform_color([0.0, 1.0, 0])
            if "target_dbscan" in O3D_VIS:
                o3d.visualization.draw_geometries(target_clusters_clouds, "simulated scan (target) clusters")
            list_target_clusters_clouds += copy.deepcopy(target_clusters_clouds)
        target_clusters_clouds = list_target_clusters_clouds
        if "multiple_sims" in O3D_VIS:
            o3d.visualization.draw_geometries(target_clusters_clouds, "multiple sims")

        

        # ################
        # # retrieve another scan with index offset-> not localization -> Scan Alignment Test
        # scan2_path = scan_paths[START_SCAN+i*SCAN_SAMPLING+SCAN2_OFFSET]
        # scan2_points, _ = read_scan(scan2_path)
        # scan2 = o3d.geometry.PointCloud()
        # scan2.points = o3d.utility.Vector3dVector(scan2_points)
        # # preprocess scan
        # scan2 = preprocess_pcd(copy.deepcopy(scan2), True, np.array([0.0,0.0,0.0]), 15)
        # scan2.transform(initial_guess)
        # scan2.paint_uniform_color([1.0, 0.0, 0])
        # o3d.visualization.draw_geometries([scan2], "scan")
        # # segment the scan
        # # ground segmentation
        # plane_model, ground_inliers = segment_ground(copy.deepcopy(scan2), o3d_vis=False)
        # ground_cloud = copy.deepcopy(scan2).select_by_index(ground_inliers)
        # outlier_cloud = copy.deepcopy(scan2).select_by_index(ground_inliers, invert=True)
        # ground_cloud.paint_uniform_color([0.0, 0.0, 0])
        # outlier_cloud.paint_uniform_color([0.0, 1.0, 0])
        # o3d.visualization.draw_geometries([ground_cloud, outlier_cloud], "scan 2: segmented plane with outlier cloud")
        # # DBSCAN
        # # segment outlier cloud using DBSCAN algorithm
        # target_clusters_clouds = segment_dbscan(copy.deepcopy(outlier_cloud), eps=0.8, min_points=20, min_cluster_points=35)
        # o3d.visualization.draw_geometries(target_clusters_clouds, "scan 2: clusters")
        # # paint with blue for visualization
        # for cluster in target_clusters_clouds:
        #     cluster.paint_uniform_color([0.0, 0.0, 1.0])     
        

        ################
        # SIMILARITY 
        source_segments = []
        for segment in source_clusters_clouds:
            source_segment = SourceSegment(copy.deepcopy(segment), np.array([0,0,0]))
            source_segments.append(source_segment)
        
        for segment in target_clusters_clouds:
            target_segment = TargetSegment(copy.deepcopy(segment), np.array([0,0,0]), 6.0, training=TRAINING)
            target_segment.get_similar_segments_candidates(source_segments) # not copy .. source segments are to be updated using each target segment
            # visualize candidates
            vis_bbs = []
            for candidate_index in range(len(target_segment.similar_candidates)):
                candidate_bb = copy.deepcopy(target_segment.similar_candidates[candidate_index].pcd).get_oriented_bounding_box()
                candidate_bb_lineset = lineset_from_oriented_bb(candidate_bb, color=[1.0, 0.0, 0.0])
                vis_bbs.append(candidate_bb_lineset)
            target_bb = copy.deepcopy(target_segment.pcd).get_oriented_bounding_box()
            target_bb_lineset = lineset_from_oriented_bb(target_bb, color=[0.0, 0.0, 0.0])
            vis_bbs.append(target_bb_lineset) 
            if "each_target_similarity_candidates" in O3D_VIS:
                o3d.visualization.draw_geometries(source_clusters_clouds+target_clusters_clouds+vis_bbs, f"target segment with similar segments candidates")
            
            # calculate feature vectors for the candidates
            x,y = target_segment.calculate_candidates_similarity_feature_vector()
            # add feature vector for training
            X += x
            Y += y
            # Predictions for a target segment similar candidates 
            if not TRAINING:
                if len(target_segment.similar_candidates_feature_vectors) > 0:
                    predictions, probabilities = CLF.predict(np.array(target_segment.similar_candidates_feature_vectors))
                    print(f"Predictions: {predictions}, Probabilities: {probabilities}")          
                    
                    ###
                    # visualize candidates classification
                    # visualize .. red -> target segment .. green -> candidate classified false .. black -> candidate classified true
                    vis_bbs = []
                    for candidate_index in range(len(target_segment.similar_candidates)):
                        candidate_bb = copy.deepcopy(target_segment.similar_candidates[candidate_index].pcd).get_oriented_bounding_box()
                        if predictions[candidate_index] < 0.5:  # 0 or 1 # 0 not similar -> Red
                            candidate_bb_lineset = lineset_from_oriented_bb(candidate_bb, color=[1.0, 0.0, 0.0])
                        else:
                            candidate_bb_lineset = lineset_from_oriented_bb(candidate_bb, color=[0.0, 1.0, 0.0])
                        vis_bbs.append(candidate_bb_lineset)
                    target_bb = copy.deepcopy(target_segment.pcd).get_oriented_bounding_box()
                    target_bb_lineset = lineset_from_oriented_bb(target_bb, color=[0.0, 0.0, 0.0])
                    vis_bbs.append(target_bb_lineset) 
                    if "each_target_similarity_candidates_results" in O3D_VIS:
                        o3d.visualization.draw_geometries(source_clusters_clouds+target_clusters_clouds+vis_bbs, f"target segment with similar segments candidates")

                    ###
                    ### Update each source segment (similarity candidate) with possible transformation and its probability
                    for candidate_index in range(len(target_segment.similar_candidates)):
                        target_segment.similar_candidates[candidate_index].add_similar_segment((probabilities[candidate_index])[1], 
                                                                                               copy.deepcopy(target_segment.pcd))

                
        ### LASTLY -> GLOBAL TRANS AND WEIGHTED ICP
        if not TRAINING:
            matches = []
            weights = []
            source = copy.deepcopy(scan)
            target = copy.deepcopy(simulated_scan)
            for i in range(len(source_segments)):
                source_segments[i].set_highest_weight()
                # weights.append(source_segments[i].highest_weight)
                if source_segments[i].highest_weight == 0 and len(source_segments[i].similar_segments)>0:
                    weights.append(0.1)
                else:
                    weights.append(source_segments[i].highest_weight)
                source_matches = copy.deepcopy(source_segments[i].similar_segments)
                for source_match in source_matches:
                    if source_match[0] > -0.0:   #CHECK # TEMP LOW DUE TO CLASSIFICATION INACCURACY
                        matches.append([source_match[0], copy.deepcopy(source_segments[i].pcd), source_match[1]])                      
            WICP = WeightedICP()
            global_trans, global_trans_weight = WICP.calculate_global_transformation(matches)
            print("global transformation weight:", global_trans_weight)
            global_trans_done = False
            global_trans_trace_value = (np.trace(global_trans[:3, :3]) - 1) / 2
            global_trans_trace_value_clamped = np.clip(global_trans_trace_value, -1.0, 1.0)
            global_trans_rot_rad = np.arccos(global_trans_trace_value_clamped)
            global_trans_rot_deg =  np.degrees(global_trans_rot_rad)  
            if global_trans_weight > 0.6:
                global_trans_done = True
                # TRANSFORM SOURCE
                source.transform(global_trans)
                for cluster in source_clusters_clouds:
                    cluster.transform(global_trans) ## DONEEE
                if "global_transformation" in O3D_VIS:
                    o3d.visualization.draw_geometries(source_clusters_clouds+target_clusters_clouds, "GLOBAL TRANSFORM")
                icp_result = WICP.weighted_icp(source_clusters_clouds, weights, target_clusters_clouds)
            else: # normal ICPs
                icp_result = o3d.pipelines.registration.registration_icp(source, target, 6.0, np.eye(4), 
                                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            source.transform(icp_result.transformation)
            if "final_result" in O3D_VIS:
                o3d.visualization.draw_geometries([source, target], "Final Result")

            # Evaluation Temp
            source_eval_icp = copy.deepcopy(scan) 
            source_eval_icp.transform(initial_guess)
            map_full = o3d.io.read_point_cloud(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/map_point_cloud.ply")))
            map_full.paint_uniform_color([0.0, 1.0, 0.0])
            target_eval_icp = preprocess_pcd(copy.deepcopy(map_full), True, initial_guess[0:3,3], 25)
            target_eval_icp.paint_uniform_color([0.0, 1.0, 0.0])
            if "icp_initial" in O3D_VIS:
                o3d.visualization.draw_geometries([source_eval_icp, target_eval_icp], "ICP INITIAL")
            if global_trans_done:
                correction_transform = icp_result.transformation @ global_trans
            else:
                correction_transform = icp_result.transformation
            ESTIMATION_OURS.append(initial_guess @ correction_transform)
            print("EST",ESTIMATION_OURS[-1])
            # do icp for compare
            icp_result_eval_icp = o3d.pipelines.registration.registration_icp(source_eval_icp, target_eval_icp, 6.0, np.eye(4), 
                                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
            source_eval_icp.transform(icp_result_eval_icp.transformation)
            if "eval_vanilla_icp" in O3D_VIS:
                o3d.visualization.draw_geometries([source_eval_icp, target_eval_icp], "Result Normal ICP")
            source_eval_gt = copy.deepcopy(scan)
            source_eval_gt.transform(GROUND_TRUTH[-1])
            print("gt",GROUND_TRUTH[-1])
            target.transform(initial_guess)
            if "eval_ground_truth" in O3D_VIS:
                o3d.visualization.draw_geometries([source_eval_gt, target_eval_icp], "Check Ground Truth")
            ESTIMATION_ICP.append(icp_result_eval_icp.transformation @initial_guess)
            if "final_result_scan_map" in O3D_VIS:
                source_eval_ours = copy.deepcopy(scan)
                source_eval_ours.transform(ESTIMATION_OURS[-1])
                o3d.visualization.draw_geometries([source_eval_ours, target_eval_icp], "EVAL OURS")

        
    if EVALUATION:
        with open(GROUND_TRUTH_SAVE_PATH, "w") as f:
            for i in range(len(GROUND_TRUTH)):
                np.savetxt(f, GROUND_TRUTH[i])
        f.close
        with open(ESTIMATION_OURS_SAVE_PATH, "w") as f:
            for i in range(len(ESTIMATION_OURS)):
                np.savetxt(f, ESTIMATION_OURS[i])
        f.close
        with open(ESTIMATION_ICP_SAVE_PATH, "w") as f:
            for i in range(len(ESTIMATION_ICP)):
                np.savetxt(f, ESTIMATION_ICP[i])
        f.close
        with open(INIT_GUESS_DEVIATION_SAVE_PATH, "w") as f:
            for i in range(len(INIT_GUESS_DEVIATION)):
                np.savetxt(f, INIT_GUESS_DEVIATION[i])
        f.close
            
    
    if TRAINING:
        X = np.array(X)
        Y = np.array(Y)
        CLF.train(X,Y)
        CLF.save_model()
        np.savetxt(TRAIN_SAVE_X_PATH, X)
        np.savetxt(TRAIN_SAVE_Y_PATH, Y)
        # visualize training data first three dimension
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=Y, cmap='viridis')
        fig.colorbar(scatter, ax=ax)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('X2')
        fig.show()
        input("Input Wait for vis")

       
       
##################################################################################################################
##################################################################################################################

# ##############
# # other approach .. full map rather simulated scan ...
# map_full_around_init = preprocess_pcd(copy.deepcopy(map_full), True, initial_guess[0:3,3], 15)
# o3d.visualization.draw_geometries([map_full_around_init], "other approach")
# clusters_full_map = segment_full_map(copy.deepcopy(map_full_around_init), plane_model)
# o3d.visualization.draw_geometries(clusters_full_map, "other approach")
# for cluster in clusters_full_map:
#     cluster.paint_uniform_color([0.0, 0.0, 0.0])
# o3d.visualization.draw_geometries(clusters_full_map+clusters_clouds, "ALL")

# # publish result
# map_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(map.points))
# scan_init_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(scan_initial_guess.points))
# scan_registered_ros = point_cloud2.create_cloud_xyz32(Header(frame_id="map"), np.asarray(scan_registered.points))
# map_pub.publish(map_ros)
# scan_init_pub.publish(scan_init_ros)
# scan_registered_pub.publish(scan_registered_ros)