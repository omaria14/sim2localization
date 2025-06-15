import numpy as np
import copy
import open3d as o3d
from sklearn.decomposition import PCA
import itertools

def compute_rotation_error(R_gt, R_est):
    R_error = R_gt @ R_est.T  # Relative rotation matrix
    # Clamp the value to be between -1 and 1 to avoid numerical errors
    trace_value = (np.trace(R_error) - 1) / 2
    trace_value_clamped = np.clip(trace_value, -1.0, 1.0)  # Clamp the value to avoid invalid arccos
    angle_error = np.arccos(trace_value_clamped)  # Rotation error in radians
    return np.degrees(angle_error)  # Convert to degrees

class WeightedICP:
    @staticmethod   
    def calculate_global_transformation(matches, threshold=1.0, # matche -> [prob., source, target]
                                        o3d_viz=[]): 
                                        # "all_matches", "best_comb", "best_supporters"
                                        # "comb_source_target", "comb_transformation"
                                        # "each_match_transformed_with_comb", comb_matches_transformation
        # source and target for debug
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        for match in matches:
            source.points.extend(match[1].points)
            target.points.extend(match[2].points)
        source.paint_uniform_color([1.0, 0.0, 0])
        target.paint_uniform_color([0.0, 1.0, 0])
        if "all_matches" in o3d_viz:
            o3d.visualization.draw_geometries([source, target], "all matches")

        ###
        # extend match to include source_alone_transformed
        for match in matches:
            icp_result = o3d.pipelines.registration.registration_icp(match[1], match[2], 10, np.eye(4), 
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5))
            source_alone_transformed = copy.deepcopy(match[1])
            source_alone_transformed.transform(icp_result.transformation)
            match.append(source_alone_transformed)
            match.append(np.mean(np.asarray(source_alone_transformed.points),axis=0)) # for position test only from mean!!!! ->
            # very important not to compare translations taken from overall transformation as it depends on the estimated rotation introdcuing errors
            # for comb ini translation -> should be translation estimated alone (not estimated within rotation rotation)!
            match.append(np.mean(np.asarray(match[2].points), axis=0) - np.mean(np.asarray(match[1].points), axis=0))
            if "each_match_alone_translation" in o3d_viz:
                source_alone_translated_debug = copy.deepcopy(match[1])
                source_alone_translated_debug.translate(match[5])
                o3d.visualization.draw_geometries([source_alone_translated_debug, match[2]], "Debug Match Source Alone Translated")
            # append the transform itself for rotation test
            match.append(icp_result.transformation)
            # match now prob0, source1, target2, source_alone_transformed3, source_alone_transformed_mean4, source_alone_ini_trans5, transform6
                        
        ###
        combinations = list(itertools.combinations(copy.deepcopy(matches), 2))
        combinations.extend((match, [0, None, None]) for match in matches)
        best_transformation = np.eye(4)
        best_score = 0
        best_score_suggestors = 0
        debug_best_comb = []
        debug_best_supporters = []
        for combination in combinations:
            current_score = 0
            current_suggestors_score = combination[1][0] + combination[1][0]
            debug_supporters = []
            if combination[1][1] is None:
                comb_source = copy.deepcopy(combination[0][1])
                comb_target = copy.deepcopy(combination[0][2])
            else: 

            # generate source from combination
                comb_source = o3d.geometry.PointCloud()
                comb_source.points.extend(copy.deepcopy(combination[0][1]).points)
                comb_source.points.extend(copy.deepcopy(combination[1][1]).points)
                # generate target from combination
                comb_target = o3d.geometry.PointCloud()
                comb_target.points.extend(copy.deepcopy(combination[0][2]).points)
                comb_target.points.extend(copy.deepcopy(combination[1][2]).points)
            comb_source.paint_uniform_color([1.0, 0.0, 0])
            comb_target.paint_uniform_color([0.0, 1.0, 0])
            if "comb" in o3d_viz:
                o3d.visualization.draw_geometries([comb_source, comb_target], "comb")
            # estimate from generated source and target a transformation
            # initial guess with translations
            comb_ini = np.eye(4)
            comb_ini[:3, 3] = (combination[0][5] + 0 if combination[1][1] is None else combination[1][5])/2
            # debug comb ini
            if "comb_ini_trans" in o3d_viz: 
                comb_source_debug = copy.deepcopy(comb_source)
                comb_source_debug.transform(comb_ini)
                o3d.visualization.draw_geometries([comb_source_debug, comb_target], "comb Ini (Translation)")
            icp_result = o3d.pipelines.registration.registration_icp(comb_source, comb_target, 4.0, comb_ini, 
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
            transformation = copy.deepcopy(icp_result.transformation)
            # debug vis
            if "comb_transformed" in o3d_viz:
                comb_source.transform(transformation)
                o3d.visualization.draw_geometries([comb_source, comb_target], "comb transform")
            # get transformation score: loop over matches and check if transforming source mean leads to source_local_transformation
            for match in matches:
                source_test = copy.deepcopy(match[1])
                source_test.transform(transformation)
                source_test_mean = np.mean(np.asarray(source_test.points), axis=0)
                # debug vis
                if "comb_each_match_transformed" in o3d_viz:
                    o3d.visualization.draw_geometries([source_test, match[2]], "comb each match treansformed")
                if np.linalg.norm(source_test_mean - match[4]) < threshold and \
                    compute_rotation_error(transformation[:3, :3], (match[6])[:3, :3]) < 15:
                    print("rotation error:",compute_rotation_error(transformation[:3, :3], (match[6])[:3, :3]))
                    # second check rotation matrix allowing error angle 40?
                    current_score += match[0]
                    debug_supporters.append(copy.deepcopy(match[1]))
                    debug_supporters[-1].paint_uniform_color([0, 0, 0])
                    debug_supporters.append(copy.deepcopy(match[2]))
                    debug_supporters[-1].paint_uniform_color([0, 0, 1])
            if "comb_matches_transformed" in o3d_viz:
                source_debug = copy.deepcopy(source)
                source_debug.transform(transformation)
                o3d.visualization.draw_geometries([source_debug, target], f"comb transformation on all matches .. score {current_score}")
            if "comb_supporters" in o3d_viz:
                o3d.visualization.draw_geometries([source]+[target]+debug_supporters, f"comb supporters .. score {current_score}")
            if current_score>best_score or (current_score==best_score and current_suggestors_score>best_score_suggestors):
                best_score = current_score
                best_transformation = transformation
                best_score_suggestors = current_suggestors_score
                debug_best_comb = [comb_source, comb_target]
                debug_best_supporters = copy.deepcopy(debug_supporters)
        if "best_comb" in o3d_viz:
            o3d.visualization.draw_geometries(debug_best_comb, f"best comb with score {best_score}")
        if "best_supporters" in o3d_viz:
            o3d.visualization.draw_geometries([source]+[target]+debug_best_supporters, f"best supporters")
        return best_transformation, best_score
 

    @staticmethod
    def weighted_icp(source_segments, weights, target_segments, weight_threshold=0.05, o3d_viz=False):  # TEMP #CHECK 
        source = o3d.geometry.PointCloud()
        source_thresholded = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        for i in range(len(target_segments)):
            target.points.extend(target_segments[i].points)
            if target_segments[i].has_colors():
                target.colors.extend(target_segments[i].colors)
            if target_segments[i].has_normals():
                target.normals.extend(target_segments[i].normals)

        for i in range(len(source_segments)):
            source.points.extend(source_segments[i].points)
            if source_segments[i].has_colors():
                source.colors.extend(source_segments[i].colors)
            if source_segments[i].has_normals():
                source.normals.extend(source_segments[i].normals)
            if weights[i] > weight_threshold:
                source_thresholded.points.extend(source_segments[i].points)
                if source_segments[i].has_colors():
                    source_thresholded.colors.extend(source_segments[i].colors)
                if source_segments[i].has_normals():
                    source_thresholded.normals.extend(source_segments[i].normals)
            
        
        if o3d_viz:
            o3d.visualization.draw_geometries([source_thresholded], "thresholded source")
        icp_result = o3d.pipelines.registration.registration_icp(
            source_thresholded, target, 1.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        source.transform(icp_result.transformation)
        if o3d_viz:
            o3d.visualization.draw_geometries([source, target], "thresholded ICP")
        return icp_result