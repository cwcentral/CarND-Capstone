from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
import cv2
import sys
import rospy
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.stats import norm
from mercurial.encoding import lower

#
# See https://github.com/udacity/CarND-Object-Detection-Lab/blob/master/CarND-Object-Detection-Lab.ipynb
# Section: Object detection Inference
#

SSD_GRAPH_FILE = 'light_classification/frozen_inference_graph.pb'

class TLClassifier(object):

    def __init__(self):
        #TODO load classifier

        self.detection_graph = self.load_graph(SSD_GRAPH_FILE)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.detection_number = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

        # See TL message
        self.light_colors = {
            1 : TrafficLight.GREEN,
            2 : TrafficLight.RED,
            3 : TrafficLight.YELLOW,
            4 : TrafficLight.UNKNOWN
        }

        #self.image_pub = rospy.Publisher("/light_image_topic",Image, queue_size=10)

        self.blank_image = None

    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def hasColor(self, img, lower, upper):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        img_after = cv2.bitwise_and(img,img, mask=mask)
        hsv_channels = cv2.split(img_after);
        ret, img_bin = cv2.threshold(hsv_channels[2], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im3, contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
       
#         try:
#             cv2.imshow("Binary Window", img_bin)
#             cv2.waitKey(3)
#         except CvBridgeError as e:
#             print(e)

        if len(contours) > 4:
            return True

        return False

    def get_color(self, img):
        lower_yellow = np.array([15, 0, 0])
        upper_yellow = np.array([36, 255, 255])
       
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([0, 255, 255])
       
        lower_green = np.array([60, 60, 60])
        upper_green = np.array([80, 255, 255])
        
        if self.hasColor(img, lower_red, upper_red):
            return TrafficLight.RED
        elif self.hasColor(img, lower_yellow, upper_yellow):
            return TrafficLight.YELLOW
        elif self.hasColor(img, lower_green, upper_green):
            return TrafficLight.GREEN
               
        return TrafficLight.UNKNOWN
           
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, 
                                                       self.detection_scores, 
                                       self.detection_classes, self.detection_number], 
                                        feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)
        idx = np.argmax(scores)

        height, width, _ = image.shape
        self.blank_image = np.zeros((height, width, 3), np.uint8)
 

        if scores is not None: 
            filtered_results = []
            for i in range(0, num):
                score = scores[i]

                # COCO Dataset has traffic lights as class id 10
                if score >= 0.45 and classes[i] == 10:
                    y1, x1, y2, x2 = boxes[i]
                    y1_o = int(y1 * height)
                    x1_o = int(x1 * width)
                    y2_o = int(y2 * height)
                    x2_o = int(x2 * width)
                    predicted_class = classes[i]
                    filtered_results.append({
                        "score": score,
                        "bb": boxes[i],
                        "bb_o": [x1_o, y1_o, x2_o, y2_o],
                        "img_size": [height, width],
                        "class": predicted_class
                    })
                    #print('[INFO] %s: %s' % (predicted_class, score))
                    cv2.rectangle(image,(x1_o,y1_o),(x2_o,y2_o),(0,255,255),2)
                    self.blank_image = image

#         try:
#             cv2.imshow("Detection Window", self.blank_image)
#             cv2.waitKey(3)
#  
#         except CvBridgeError as e:
#             print(e)

        if len(filtered_results) > 0:
            x1_o = filtered_results[0]["bb_o"][0] 
            y1_o = filtered_results[0]["bb_o"][1]
            x2_o = filtered_results[0]["bb_o"][2]
            y2_o = filtered_results[0]["bb_o"][3]
            roi = self.blank_image[y1_o:y2_o, x1_o:x2_o]
            # Use lesson on histograms from CarND-Advanced-Lane-Lines
            return self.get_color(roi)
            
#         for obj in filtered_results:
#             x1_o = obj["bb_o"][0] 
#             y1_o = obj["bb_o"][1]
#             x2_o = obj["bb_o"][2]
#             y2_o = obj["bb_o"][3]
#             roi = self.blank_image[y1_o:y2_o, x1_o:x2_o]
#             # Use lesson on histograms from CarND-Advanced-Lane-Lines
#             return self.get_color(roi)
            
        

        return TrafficLight.UNKNOWN

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
    
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords
