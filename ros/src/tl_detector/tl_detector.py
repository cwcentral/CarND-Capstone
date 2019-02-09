#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from scipy.spatial import KDTree
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 2

# When GPU enable --  chaneg this to 2
#image_rate = 2
# STYX NON GPU
image_rate = 5 
#image_rate = 20 

# counter
image_rate_ctn = 0

# test mode . set to FALSE if using a BAG file
USE_WAYPOINT_PUBLISHER = True

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.last_light_state = TrafficLight.UNKNOWN

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        # create KDtree here too
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        global image_rate
        global image_rate_ctn
        
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        if (image_rate_ctn < 30) or ((image_rate_ctn % image_rate) == 0):
            
            if not USE_WAYPOINT_PUBLISHER:
                print('no planner')
                state = self.process_traffic_light_image()
            else:
                light_wp, state = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if USE_WAYPOINT_PUBLISHER:
                try:
                    if self.state != state:
                        self.state_count = 0
                        self.state = state
                    elif self.state_count >= STATE_COUNT_THRESHOLD:
                        self.last_state = self.state
                        light_wp = light_wp if state == TrafficLight.RED else -1
                        self.last_wp = light_wp
                        self.upcoming_red_light_pub.publish(Int32(light_wp))
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                    self.state_count += 1
                except:
                    print("invalid state")

        image_rate_ctn = image_rate_ctn + 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return 0

    def process_traffic_light_image(self):
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #Get classification
        sstate = self.light_classifier.get_classification(cv_image)
        print("[INFO] Light State: ", sstate)
        return sstate


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        closest_light = None
        line_wp_idx = None
        local_state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        try:
            stop_line_positions = self.config['stop_line_positions']
            if(self.pose):
                #TODO find the closest visible traffic light (if one exists)
                car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,
                                                   self.pose.pose.position.y)

                diff = len(self.waypoints.waypoints)
                for i, light in enumerate(self.lights):
                    line = stop_line_positions[i]
                    temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                    d = temp_wp_idx - car_wp_idx
                    if d >= 0 and d < diff:
                       diff = d
                       closest_light = light
                       line_wp_idx = temp_wp_idx

            if closest_light:
                # process true image
                local_state = self.get_light_state(closest_light)
                print("[INFO] Light State: ", local_state)
                return line_wp_idx, local_state
    
            #self.waypoints = None

        except NameError:
            print('Waypoints not loaded, please check base_waypoints topic in ROS')

        print("[INFO] Light State: ", local_state)
        return -1, local_state 

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
