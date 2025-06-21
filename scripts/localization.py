import numpy as np
import copy
import open3d as o3d
from sklearn.decomposition import PCA
import itertools
import tf.transformations
from scipy.spatial.transform import Rotation as R



class Localizer:
    """
    Provides collection of static functions for localization.
    """
    @staticmethod
    def calculate_global_transformation(matches, source_segments, o3d_viz=False):
        """
        A function that estimates a global transformation between source and target scans.
        Given possible matches between source segments and target segments, a transformation 
        is calculated from each match and the best transformation is decided based on 
        sum of similarity probabilities of segments that support the estimated transformation.

        Args:
            matches(list): list of matches, each match is in the form 
                [(source_segment, targetsegment), (source_segment, targetsegment), ...]
            source_segments(list): list of SourceSegment objects.
            o3d_viz(bool): if True, o3d visualization is active
        
        Returns:
            best_transformation(np.ndarray): estimated global transformation
            best_score(float): score associated with best_transformation  
        """
        print("START GLOBAL TRANSFORMATION")
        best_tranformation = np.eye(4)
        best_score = -1
        best_distances = 0
        debug_rot = -1
        debug_rot_euler = []
        for match in matches:
            # create a source and target clouds from the match
            match_score = 0
            distances = 0
            match_source = o3d.geometry.PointCloud()
            match_target = o3d.geometry.PointCloud()
            for i in range(len(match)):
                match_source.points.extend(match[i][0].pcd_points)
                match_target.points.extend(match[i][1].pcd_points)
            # estimate a transformation
            icp_result = o3d.pipelines.registration.registration_icp(match_source, match_target, 10, np.eye(4), 
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5))
            # check: limit transformation angles
            rot_matrix = copy.deepcopy(icp_result.transformation[:3,:3])
            rot = R.from_matrix(rot_matrix)
            rotvec = rot.as_rotvec()
            angle = np.linalg.norm(rotvec)
            if (angle>0.3):
                continue
            ax, ay, az = tf.transformations.euler_from_matrix(icp_result.transformation[:3,:3])
            if (ax>0.3 or ay>0.3 or az>0.3):
                continue
             
            if o3d_viz:
                match_source.paint_uniform_color([1.0, 0.0, 0])
                match_target.paint_uniform_color([0.0, 1.0, 0])
                match_source_debug = copy.deepcopy(match_source)
                match_source_debug.transform(icp_result.transformation)
                o3d.visualization.draw_geometries([match_source_debug, match_target], f"suggested match")
            
            # calculate the score by applying the trans on source_segments
            # score s calculated by adding similarity probabilities of target candidates that
            # become closer after the transformation
            target_supporters = []
            source_supporters = []
            
            for source_segment in source_segments:
                # calculate the mean and see if the distance between mean and target candidates 
                # gets closer than threshold -> add similarity probab to score
                source_segment_transformed_pcd = copy.deepcopy(source_segment.pcd)
                source_segment_transformed_pcd.transform(icp_result.transformation)
                source_segment_transformed_mean = np.mean(np.asarray(source_segment_transformed_pcd.points))
                
                for i in range(len(source_segment.similar_candidates)):
                    candidate_target_segment_mean = source_segment.similar_candidates[i].mean
                    distance = np.linalg.norm(source_segment_transformed_mean - candidate_target_segment_mean)
                    if distance < 0.8:
                        match_score += \
                        source_segment.similar_candidates_classification_results[i][1]
                        if o3d_viz:
                            target_supporter = copy.deepcopy(source_segment.similar_candidates[i].pcd)
                            target_supporter.paint_uniform_color([0.0, 1.0, 0.0])
                            target_supporters.append(target_supporter)
                            source_supporter = copy.deepcopy(source_segment.pcd)
                            source_supporter.paint_uniform_color([1.0, 0.0, 0.0])
                            source_supporters.append(source_supporter)
                            
            if o3d_viz:
                o3d.visualization.draw_geometries(source_supporters+target_supporters, f"supporters score:{match_score}")
            
            if match_score > best_score or \
                (match_score==best_score and distances<best_distances):
                best_distances = distances
                best_score = match_score
                best_tranformation = copy.deepcopy(icp_result.transformation)
                debug_rot = angle
                debug_rot_euler = [ax, ay, az]

        print(debug_rot, debug_rot_euler)
        print("END GLOBAL TRANSFORMATION")
        return  best_tranformation, best_score
                                        
 

    @staticmethod
    def thresholded_icp(source_segments, target_segments, 
                        initial_transformation= np.eye(4),ICP_THRESHOLD=0.1, o3d_viz=False):
        """
        A function that applies ICP considering only stable source segments.

        Args:
            source_segments(list): list of SourceSegment objects
            target_segments(list): list of TargetSegment objects
            initial_transformation(np.ndarray): estimated global transformation
            ICP_THRESHOLD(float): minimal similarity threshold that a source_segment.similar_candidates
                should achieve in order to be included in the ICP refinement.
        
        Returns:
            icp_result(o3d.pipelines.registration.RegistrationResult): ICP result
        """
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        for i in range(len(source_segments)):
            source_segments[i].set_highest_score()
            if source_segments[i].highest_score> ICP_THRESHOLD:
                source.points.extend(source_segments[i].pcd_points)
                if source_segments[i].pcd.has_colors():
                    source.colors.extend(source_segments[i].pcd.colors)
                if source_segments[i].pcd.has_normals():
                    source.normals.extend(source_segments[i].pcd.normals)

        for i in range(len(target_segments)):
            target.points.extend(target_segments[i].pcd_points)
            if target_segments[i].pcd.has_colors():
                target.colors.extend(target_segments[i].pcd.colors)
            if target_segments[i].pcd.has_normals():
                target.normals.extend(target_segments[i].pcd.normals)
            
        
        if o3d_viz:
            o3d.visualization.draw_geometries([source], "thresholded source")
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, 1.0, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        source.transform(icp_result.transformation)
        if o3d_viz:
            o3d.visualization.draw_geometries([source, target], "thresholded ICP")
        return icp_result