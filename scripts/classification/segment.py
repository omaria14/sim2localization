#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import copy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class Segment:
    def __init__(self, pcd, camera_location) -> None:
        self.pcd = pcd
        self.pcd_points = np.asarray(pcd.points)
        self.camera_location = camera_location
        self.set_dominant_range_mean()
        self.set_principal_dim()

    
    def set_dominant_range_mean(self, percentage=0.80):
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


class SourceSegment(Segment):
    def __init__(self, pcd, camera_location) -> None:
        super().__init__(pcd, camera_location)
        self.similar_segments = []
        self.highest_weight = 0
        
    def add_similar_segment(self, weight, target_cloud):
        self.similar_segments.append((weight, target_cloud))
    
    def set_highest_weight(self):
        if len(self.similar_segments) < 1:
            return
        max_sublist = max(self.similar_segments, key=lambda x: x[0])
        max_weight = max_sublist[0]
        self.highes_weight = max_weight
        return




class TargetSegment(Segment):
    def __init__(self, pcd, camera_location, similarity_search_offset, training=False) -> None:
        super().__init__(pcd, camera_location)
        self.set_size_class()
        self.similarity_search_offset = similarity_search_offset
        self.similar_candidates = []
        self.similar_candidates_feature_vectors = []
        
        # for training ..
        self.training = training
        self.similar_candidates_labels = []
    
    def set_size_class(self, length_threshold=2.0, width_threshold=1.5):
        if self.length > length_threshold and self.width > width_threshold:
            self.size_class = "large"
        else:
            self.size_class = "small"
        print(f"Size Class: {self.size_class}, Length: {self.length}, Width: {self.width}")
            
    
    def get_similar_segments_candidates(self, lst_other_segments):
        for segment in lst_other_segments:
        
            x1_mean, y1_mean, z1_mean = self.dominant_range_mean
            x2_mean, y2_mean, z2_mean = segment.dominant_range_mean

            x_check = (x1_mean + self.similarity_search_offset >= x2_mean and  x1_mean - self.similarity_search_offset <= x2_mean)
            y_check = (y1_mean + self.similarity_search_offset >= y2_mean and  y1_mean - self.similarity_search_offset <= y2_mean)
            z_check = (z1_mean + self.similarity_search_offset >= z2_mean and  z1_mean - self.similarity_search_offset <= z2_mean)

            if x_check and y_check and z_check:
                self.similar_candidates.append(segment)
    
    def calculate_candidates_similarity_feature_vector(self):
        Y = []  # labels to be used along self.similar_candidates_feature_vectors if self.training!
        # ensure no old data .. 
        self.similar_candidates_feature_vectors = []
        self.similar_candidates_labels = []
        
        for i in range(len(self.similar_candidates)):
            H1 = self.get_histogram_of_normals(self.pcd, camera_location=self.camera_location)
            H2 = self.get_histogram_of_normals(self.similar_candidates[i].pcd, camera_location=self.similar_candidates[i].camera_location)
            hist_difference = np.linalg.norm(H2-H1)
            # feature vector is hist difference norm and area of the segment .. as the norm could change with segment area ..
            self.similar_candidates_feature_vectors.append([hist_difference, self.length, self.width,
                                                            min(self.length,self.similar_candidates[i].length)/max(self.length,self.similar_candidates[i].length),
                                                            min(self.length,self.similar_candidates[i].width)/max(self.length,self.similar_candidates[i].width)]) 

            if self.training:
                o3d.visualization.draw_geometries([self.pcd, self.similar_candidates[i].pcd], f"Choose a Label! - Size Class: {self.size_class}")
                label = input("Enter The Label, 1 -> Similar, 0 -> Not Similar, S -> Skip")
                if label != "0" and label != "1":
                    self.similar_candidates_feature_vectors.pop()
                else:
                    Y.append(float(label))
                    print("Label:", Y[-1], " Feature Vector: ",  self.similar_candidates_feature_vectors[-1])
        return self.similar_candidates_feature_vectors,Y  # return feature vector and labels 

    
