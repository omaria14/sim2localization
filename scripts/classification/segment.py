#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class Segment:
    """
    Considers a Point Cloud Segment, implement basic methods and saves required member variables.
    """
    def __init__(self, id, pcd, camera_location) -> None:
        self.id = id
        self.pcd = pcd
        self.pcd_points = np.asarray(pcd.points)
        self.camera_location = camera_location
        self.set_dominant_range_mean()
        self.set_principal_dim()
        self.mean = np.mean(self.pcd_points)

    
    def set_dominant_range_mean(self, percentage=0.80):
        """
        Sets the mean of the segment point cloud considering a percentage of the points
        along X,Y,Z axes to avoid outliers effect on the mean.

        Args:
            percentage(float): percentage of points to be considered.
        """
        num_points = self.pcd_points.shape[0]
        num_included_points = int(np.floor(percentage * num_points))
        # Find ranges for each axis
        x_vals = np.sort(copy.deepcopy(self.pcd_points)[:, 0])
        y_vals = np.sort(copy.deepcopy(self.pcd_points)[:, 1])
        z_vals = np.sort(copy.deepcopy(self.pcd_points)[:, 2])
        # Determine index ranges to capture the required percentage of points
        lower_idx = (num_points - num_included_points) // 2
        upper_idx = lower_idx + num_included_points
        # Get the min/max values that contain the desired percentage of points
        x_min, x_max = x_vals[lower_idx], x_vals[upper_idx - 1]
        y_min, y_max = y_vals[lower_idx], y_vals[upper_idx - 1]
        z_min, z_max = z_vals[lower_idx], z_vals[upper_idx - 1]
        self.dominant_range_mean =  [(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
        return True

    def set_principal_dim(self, percentage=0.80):
        """
        Calculates the segment length, width and height along its principal axes. Considers only
        percentage of the points to avoid outliers effect.

        Args:
            percentage(float): percentage of points to be considered.
        """
        pca = PCA(n_components=3)
        pca.fit(self.pcd_points)
        transformed_points = pca.transform(copy.deepcopy(self.pcd_points))
        num_points = self.pcd_points.shape[0]
        num_included_points = int(np.floor(percentage * num_points))
        axis1_vals = np.sort(transformed_points[:, 0])  # Along principal axis 1 (length)
        axis2_vals = np.sort(transformed_points[:, 1])  # Along principal axis 2 (width)
        axis3_vals = np.sort(transformed_points[:, 2])  # Along principal axis 3 (height)
        lower_idx = (num_points - num_included_points) // 2
        upper_idx = lower_idx + num_included_points
        length = axis1_vals[upper_idx - 1] - axis1_vals[lower_idx]
        width = axis2_vals[upper_idx - 1] - axis2_vals[lower_idx]
        height = axis3_vals[upper_idx - 1] - axis3_vals[lower_idx]

        self.length = length
        self.width = width
        self.height = height
        return True

    @staticmethod
    def get_histogram_of_normals(pcd, camera_location, bins=6):
        """
        Calculates a global feature for the segment based on normals. The feature is a histogram of
        angles between local normals and the mean direction of all normals.
        @todo make angles between local normals and a dominant direction of the local normals
        (direction with highest number of local normal pointing to it)(not the mean).

        Args:
        camera_location(list[float]): location to orient normals
        pins(int): number of pins in the histogram

        Returns:
            angle_histogram(numpy.ndarray): resultant histogram as a global feature
        """
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(camera_location=camera_location)
        pcd.normalize_normals()
        # o3d.visualization.draw_geometries([pcd], "Segment Before Hist")
        normals = np.asarray(pcd.normals)
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        dot_products = np.dot(normals, mean_normal)
        dot_products = np.clip(dot_products, -1.0, 1.0)  # To avoid numerical errors leading to values slightly out of [-1, 1]
        angles = np.arccos(dot_products)  # Returns angles in radians between 0, pi
        bin_edges = np.linspace(0, np.pi, bins + 1)
        angle_histogram, _ = np.histogram(angles, bins=bin_edges)
        angle_histogram = angle_histogram/np.sum(angle_histogram)
        
        # # Temp Visualize PLT:
        # y = input("Visualize?")
        # if y == "y":
        #     bin_width = bin_edges[1] - bin_edges[0]
        #     plt.figure(figsize=(10, 6))
        #     plt.bar(bin_edges[:-1], angle_histogram, width=bin_width, alpha=0.7, color='blue', edgecolor='black')
        #     bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        #     plt.xticks(bin_centers - (bin_width/2), bin_labels)
        #     plt.title('Histogram of Angles between Normals and Normals Mean')
        #     plt.xlabel('Angles')
        #     plt.ylabel('Normalized Frequence')
        #     plt.grid(axis='y', alpha=0.75)
        #     plt.show()
        #     input("Enter")
        return angle_histogram




class TargetSegment(Segment):
    """
    Inherits from Segment.
    """
    def __init__(self, id, pcd, camera_location) -> None:
        super().__init__(id, pcd, camera_location)


class SourceSegment(Segment):
    """
    Inherits from Segment. Adds some needed member variables and methods for source segments.
    """
    def __init__(self, id, pcd, camera_location, similarity_search_offset) -> None:
        super().__init__(id, pcd, camera_location)
        self.similarity_search_offset = similarity_search_offset
        self.similar_candidates = []
        self.similar_candidates_feature_vectors = []
        self.similar_candidates_classification_results = [] 
        # probabilities [[not_sinimlar_prob, similar_prob],...] as the classes are ints:
        #  0 -> not similar , 1 -> similar .. they are ordered automatically
        self.highest_score = 0
        
    def get_similar_segments_candidates(self, lst_target_segments):
        """
        Gets a list of nearby target segments and sets them as a similar candidates to the 
        self source segment in the map.

        Args:
            lst_target_segments(list[TargetSegment]): list of target segments
        """
        for segment in lst_target_segments:
        
            x1_mean, y1_mean, z1_mean = self.dominant_range_mean
            x2_mean, y2_mean, z2_mean = segment.dominant_range_mean

            x_check = (x1_mean + self.similarity_search_offset >= x2_mean and  x1_mean - self.similarity_search_offset <= x2_mean)
            y_check = (y1_mean + self.similarity_search_offset >= y2_mean and  y1_mean - self.similarity_search_offset <= y2_mean)
            z_check = (z1_mean + self.similarity_search_offset >= z2_mean and  z1_mean - self.similarity_search_offset <= z2_mean)

            if x_check and y_check and z_check:
                self.similar_candidates.append(segment)
        
    def calculate_candidates_similarity_feature_vector(self, TRAINING=False):
        """
        Calculates a feature vector for each candidate target segment that encodes a similarity
        between the self source segment and that target segment. Also provides interaction for 
        training; returning labels and feature vectors to train a similarity classifier.

        Args:
            TRAINING(bool): If True, visualize the segments with o3d and allow labeling.

        Returns:
            self.similar_candidates_feature_vectors(list[list[float]]): list of feature vectors
            Y(list[int]): corresponding labels; 0->not similar, 1->similar

        """
        Y = []  # labels to be used along self.similar_candidates_feature_vectors if self.training!
        # ensure no old data .. 
        self.similar_candidates_feature_vectors = []
        self.similar_candidates_classification_results = []
        
        for i in range(len(self.similar_candidates)):
            H1 = self.get_histogram_of_normals(self.pcd, camera_location=self.camera_location)
            H2 = self.get_histogram_of_normals(self.similar_candidates[i].pcd, camera_location=self.similar_candidates[i].camera_location)
            hist_difference = np.linalg.norm(H2-H1)
            # feature vector is hist difference norm and area of the segment .. as the norm could change with segment area ..
            self.similar_candidates_feature_vectors.append([hist_difference, self.length, self.width,
                                                            min(self.length,self.similar_candidates[i].length)/max(self.length,self.similar_candidates[i].length),
                                                            min(self.length,self.similar_candidates[i].width)/max(self.length,self.similar_candidates[i].width)]) 

            if TRAINING:
                o3d.visualization.draw_geometries([self.pcd, self.similar_candidates[i].pcd], f"Choose a Label!")
                label = input("Enter The Label, 1 -> Similar, 0 -> Not Similar, S -> Skip")
                if label != "0" and label != "1":
                    self.similar_candidates_feature_vectors.pop()
                else:
                    Y.append(float(label))
                    print("Label:", Y[-1], " Feature Vector: ",  self.similar_candidates_feature_vectors[-1])
        return self.similar_candidates_feature_vectors,Y  # return feature vector and labels
    
    def set_highest_score(self):
        """
        Sets a highest similarity probability among existing similar candidates.
        """
        for i in range(len(self.similar_candidates_classification_results)):
            if self.similar_candidates_classification_results[i][1]>self.highest_score:
                self.highest_score = self.similar_candidates_classification_results[i][1]