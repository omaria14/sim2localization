import numpy as np
import matplotlib.pyplot as plt
import os

# Load the ground truth and estimation matrices
ground_truth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/evaluation/gt_690_1_100_20.txt"))
estimation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/evaluation/id_690_1_100_20.txt"))
ground_truth_matrix = np.loadtxt(ground_truth_path).reshape(-1, 4, 4)
estimation_matrix = np.loadtxt(estimation_path).reshape(-1, 4, 4)

# Ensure same number of matrices
assert ground_truth_matrix.shape == estimation_matrix.shape, "Mismatch in matrix dimensions."

# Function to compute rotation error in degrees
def compute_rotation_error(R_gt, R_est):
    R_error = R_gt @ R_est.T  # Relative rotation matrix
    # Clamp the value to be between -1 and 1 to avoid numerical errors
    trace_value = (np.trace(R_error) - 1) / 2
    trace_value_clamped = np.clip(trace_value, -1.0, 1.0)  # Clamp the value to avoid invalid arccos
    angle_error = np.arccos(trace_value_clamped)  # Rotation error in radians
    return np.degrees(angle_error)  # Convert to degrees

# Lists to store errors
translation_errors = []
rotation_errors = []

# Loop through each frame
for i in range(ground_truth_matrix.shape[0]):
    T_gt = ground_truth_matrix[i]
    print(T_gt)
    T_est = estimation_matrix[i]
    print(T_est)
    
    # Extract translation vectors (last column of the 4x4 matrix)
    translation_gt = T_gt[:3, 3]
    translation_gt[2] = 0
    translation_est = T_est[:3, 3]
    translation_est[2] = 0
    
    # Compute translation error (Euclidean distance)
    translation_error = np.linalg.norm(translation_gt - translation_est)
    translation_errors.append(translation_error)
    
    # Extract rotation matrices (top-left 3x3 submatrix)
    R_gt = T_gt[:3, :3]
    R_est = T_est[:3, :3]
    
    # Compute rotation error
    rotation_error = compute_rotation_error(R_gt, R_est)
    rotation_errors.append(rotation_error)

# Compute mean and standard deviation of errors
mean_translation_error = np.mean(translation_errors)
std_translation_error = np.std(translation_errors)
mean_rotation_error = np.mean(rotation_errors)
std_rotation_error = np.std(rotation_errors)

print(f"Mean Translation Error: {mean_translation_error:.4f} meters")
print(f"Mean Rotation Error: {mean_rotation_error:.4f} degrees")

# Plot histograms with new color scheme and fewer bins
plt.figure(figsize=(6, 3))
bins = 13

# Translation error histogram
plt.subplot(1, 2, 1)
plt.hist(translation_errors, bins, color='lightslategray', alpha=0.8, edgecolor='black', linewidth=1.2)
plt.title('Translation Errors', fontsize=10)
plt.xlabel('Error (meters)', fontsize=9)
plt.ylabel('Frequency', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Calculate mean and sigma for translation
mean_translation_error = np.mean(translation_errors)
std_translation_error = np.std(translation_errors)
print(f"translation std.dev. {std_translation_error}")
print(f"rotation std.dev. {std_rotation_error}")

# # Add mean and sigma text
# plt.text(0.95, 0.95, f'Mean: {mean_translation_error:.2f} m\n1σ: {std_translation_error:.2f} m',
#          transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
#          bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Rotation error histogram
plt.subplot(1, 2, 2)
plt.hist(rotation_errors, bins, color='teal', alpha=0.8, edgecolor='black', linewidth=1.2)
plt.title('Rotation Errors', fontsize=10)
plt.xlabel('Error (degrees)', fontsize=9)
plt.ylabel('Frequency', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Calculate mean and sigma for rotation
mean_rotation_error = np.mean(rotation_errors)
std_rotation_error = np.std(rotation_errors)

# # Add mean and sigma text
# plt.text(0.95, 0.95, f'Mean: {mean_rotation_error:.2f}°\n1σ: {std_rotation_error:.2f}°',
#          transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
#          bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# # Adjust layout with a bit more space between subplots
plt.tight_layout(pad=1.5)  # Increased padding for better spacing without shrinking plot

# Show plot
plt.show()
