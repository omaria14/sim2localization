import numpy as np
import tf.transformations
from geometry_msgs.msg import Pose

def sample_poses(pose, num_samples, posistion_variance, yaw_variance):
    """
    parameters:
        pose: list [x,y,z,q0,q1,q2,q3]
        position_variance : list [x_variance, y_variance, z_variance]
        yaw_variance: float ->  variance radians
    returns:
        list of the pose [x,y,z,q0,q1,q2,q3]
    """
    x = pose[0]
    y = pose[1]
    z = pose[2]

    euler = tf.transformations.euler_from_quaternion(pose[3:])
    yaw = euler[2]

    samples = []
    for _ in range(num_samples):
        dx = np.random.uniform(-posistion_variance[0], posistion_variance[0])
        dy = np.random.uniform(-posistion_variance[1], posistion_variance[1])
        dz = np.random.uniform(-posistion_variance[2], posistion_variance[2])
        dyaw = np.random.uniform(-yaw_variance, yaw_variance)

        sampled_yaw = yaw+dyaw
        sampled_quaternion = tf.transformations.quaternion_from_euler(euler[0], euler[1], sampled_yaw)
        sampled_pose = [x+dx, y+dy, z+dz, sampled_quaternion[0], sampled_quaternion[1], sampled_quaternion[2], sampled_quaternion[3]]
        samples.append(sampled_pose)
    return samples

def pose_msg_from_list(input):
    output = Pose()
    output.position.x = input[0]
    output.position.y = input[1]
    output.position.z = input[2]
    output.orientation.x = input[3]
    output.orientation.y = input[4]
    output.orientation.z = input[5]
    output.orientation.w = input[6]
    return output

def transformation_matrix_from_pose_list(input):
    """
    input: pose list [x,y,z,q0,q1,q2,q3]
    output: corresponding transformation matrix
    """
    trans = tf.transformations.quaternion_matrix(input[3:])
    trans[0,3] = input[0]
    trans[1,3] = input[1]
    trans[2,3] = input[2]
    return trans