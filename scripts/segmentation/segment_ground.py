#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree


def segment_ground(pcd_input, distance_threshold=0.15, min_vert_ang=30, o3d_vis=True):
    """
    min_vert_angle: minimum angle between normal of the segmented plane and z axis to consider 
        the plane as an almost-vertical plane.
    """
    pcd = copy.deepcopy(pcd_input)
    ground_found = False

    while not ground_found:
        # RANSAC plane segmentation
        plane_model, plane_inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=300)
        plane_cloud = copy.deepcopy(pcd).select_by_index(plane_inliers)
        outlier_cloud = copy.deepcopy(pcd).select_by_index(plane_inliers, invert=True)
        if o3d_vis:
            plane_cloud.paint_uniform_color([0.0, 0.0, 0])
            outlier_cloud.paint_uniform_color([0.0, 1.0, 0])
            o3d.visualization.draw_geometries([plane_cloud], "segmented plane")
            o3d.visualization.draw_geometries([plane_cloud, outlier_cloud], "segmented plane with outlier cloud")        

        # classify the plane to be almost vertical or almost horizontal according to plane normal direction
        # case almost vertical -> not a ground -> remove points with high z values and segment again
        # case almost horizontal -> other checks to be a ground
        # almost vertical -> angle between plane normal and z axis is smaller than a threshold- 
        normal_vec = np.array(plane_model[:3])/np.linalg.norm(np.array(plane_model[:3]))
        angle_with_z = np.rad2deg(np.arccos(np.dot(normal_vec, np.array([0.0,0.0,1.0]))))
        # preprocess angle with z
        if angle_with_z<0:
            angle_with_z += 360
        angle_with_z = angle_with_z%180 
        print(f"angle between segmented plane and z axis is: {angle_with_z}")
        if angle_with_z>min_vert_ang:
            print("plane is almost vertical - not a ground - update points and segment plane again")
            # remove points with high z and apply plane segmentation again
            points = np.asarray(copy.deepcopy(pcd).points)
            plane_points = np.asarray(copy.deepcopy(plane_cloud).points)
            preserve_height = np.mean(plane_points[:,2]) + 0.5*np.std(plane_points[:,2])
            for i in range(points.shape[0]):
                if (i in plane_inliers) and (points[i])[2]> preserve_height:
                    (points[i])[0] = np.nan
                    (points[i])[1] = np.nan
                    (points[i])[2] = np.nan
            pcd.points = o3d.utility.Vector3dVector(points)
            continue
        else:
            # else -> could be ground -> projection check!
            # project points onto a plane
            points = np.asarray(copy.deepcopy(pcd).points)
            points_projected = np.zeros_like(points)
            distances = []
            for i in range(points.shape[0]):  # @todo .. matrix multiplication?
                displacement_magnitude = np.dot(points[i], normal_vec) + plane_model[3]  # normal_vec should be normalized
                points_projected[i] = points[i] - displacement_magnitude * normal_vec           
                distances.append(np.linalg.norm(points_projected[i] - points[i]))
            pcd_projected = o3d.geometry.PointCloud()
            pcd_projected.points = o3d.utility.Vector3dVector(points_projected)
            pcd_projected.paint_uniform_color([0.0, 0.0, 0.0])
            if o3d_vis:
                o3d.visualization.draw_geometries([pcd_projected], "projected cloud")

            # get perpendicular vector -> point - projection
            points = np.asarray(copy.deepcopy(pcd).points)
            perps = points_projected - points
            
            # as perps would be used for sign check -> we do not want the inliers to be used in the sign check
            # overwrite them with np.nan
            perps[plane_inliers, :] = np.array([np.nan, np.nan, np.nan]) # threshold for distances from the plane to be treated as inlier

            # get sign direction with axis -> sign indicates whether a point lies in the side of ground wrt the segmented plane
            # if number of points lies outside exceeds allowed outliers -> not a ground! 
            # -> update the cloud to
            sign_check_res = []
            verts = np.zeros_like(points)
            verts[:] = np.array([0.0, 0.0, 1.0])    
            for i in range(verts.shape[0]):
                sign_check_res.append(np.dot(np.array([0.0, 0.0, 1.0]), perps[i]))
            
            sign_check_res = np.array(sign_check_res)
            cloud_under_points = points[sign_check_res>0]
            cloud_upper_points = points[sign_check_res<0]
            cloud_under = o3d.geometry.PointCloud()
            cloud_under.points = o3d.utility.Vector3dVector(cloud_under_points)
            cloud_upper = o3d.geometry.PointCloud()
            cloud_upper.points = o3d.utility.Vector3dVector(cloud_upper_points)
            cloud_under.paint_uniform_color([0.0, 0.0, 1.0])
            cloud_upper.paint_uniform_color([0.0, 1.0, 0.0])
            pcd_projected.paint_uniform_color([0.0, 0.0, 0.0])
            if o3d_vis:
                o3d.visualization.draw_geometries([cloud_upper, cloud_under], "sign check result")
            
            if (sign_check_res>0).sum()<400:
                ground_found = True
            else:
                print("number of points under the plane", (sign_check_res>0).sum())
                # update cloud -> remove inliers with check sign (-) and large distances
                underlying_points_indices = np.argwhere(sign_check_res>0)
                underlying_points_distances = np.take(distances, underlying_points_indices)  # corresponding distances
                dist_sorted_underlying_points = list(x for _, x in sorted(zip(list(underlying_points_distances), list(underlying_points_indices)), reverse=True))
                # remove points around the projection of those points
                points = np.asarray(copy.deepcopy(pcd).points)
                removed_points = []
                i = 0
                kdtree = cKDTree(points)
                while len(removed_points) < int (0.1*(len(dist_sorted_underlying_points))) and i < len(dist_sorted_underlying_points):
                    index = dist_sorted_underlying_points[i][0]
                    projection = points_projected[index]               
                    to_remove_indices = kdtree.query_ball_point(projection, 0.6)
                    for j in range(len(to_remove_indices)):
                        removed_points.append(copy.deepcopy(points[to_remove_indices[j]]))
                        points[to_remove_indices[j]][0] = np.nan
                        points[to_remove_indices[j]][1] = np.nan
                        points[to_remove_indices[j]][2] = np.nan
                    i += 1
                print("length of removed points:", len(removed_points))
                # update cloud
                if len(removed_points) > 0:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    # visualize the update
                    removed_cloud = o3d.geometry.PointCloud()
                    removed_cloud.points = o3d.utility.Vector3dVector(np.array(removed_points))
                    removed_cloud.paint_uniform_color([0.0, 0.0, 0.0])
                    pcd.paint_uniform_color([0.0, 1.0, 0.0])
                    if o3d_vis:
                        o3d.visualization.draw_geometries([pcd, removed_cloud], "removed points")
                else:
                    print("WARNING: could not update the cloud!")
    print(plane_model)
    return plane_model, plane_inliers