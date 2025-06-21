import rospy
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import ModelState
from sim2localization.srv import GetSimulatedScan
from sim2localization.srv import TransformSensor
import tf

class SimulatedScansHandler:
    def __init__(self, simulated_scans_topic='/velodyne_points', sensor_model_name='velodyne') -> None:
        self.sensor_model_name = sensor_model_name
        self.simulated_scans_subscriber = rospy.Subscriber(simulated_scans_topic, PointCloud2, 
                                                           self.simulated_scan_cb, queue_size=1)
        self.simulated_scan = PointCloud2()
        # Services
        self.get_sensor_state_service = rospy.Service("get_simulated_scan",GetSimulatedScan,
                                                       self.get_simulated_scan)
        self.transform_sensor_service = rospy.Service("transform_sensor", TransformSensor,
                                                       self.transform_sensor)
        # Clients
        self.set_sensor_state_proxy = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    
    def simulated_scan_cb(self, msg):
        self.simulated_scan = msg
    
    def get_simulated_scan(self, msg):
        return self.simulated_scan
    
    def transform_sensor(self, pose):
        # transform the gazebo senso model
        sensor_state = ModelState()
        sensor_state.model_name = self.sensor_model_name
        sensor_state.pose.position = pose.pose.position
        sensor_state.pose.orientation = pose.pose.orientation
        # string reference_frame      # set pose/twist relative to the frame of this entity (Body/Model)
                                      # leave empty or "world" or "map" defaults to world-frame
        sensor_state.reference_frame = "map" 
        self.set_sensor_state_proxy(sensor_state)
        # @todo: set tf transformation between the map and the sensor to have a full tf tree.



        return True

if __name__ == "__main__":
    rospy.init_node('simulated_scans_handler')
    _ = SimulatedScansHandler()
    rospy.spin()

    
