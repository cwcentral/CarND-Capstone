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

    def get_color_image(self, img, lower, upper):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, np.array(lower, dtype = "uint8"), np.array(upper, dtype = "uint8"))
        out_img = cv2.bitwise_and(img, img, mask = mask)
        return out_img

    def to_binary_sobel(self, img, 
                        s_thresh=(175,255), 
                        sx_thresh=(35, 100), 
                        gray_thresh=[220, 255],
                        upper=[10, 255, 255],
                        lower=[179, 255, 255]):
        img = np.copy(img)
        
        # get yellow channel
        yellow = self.get_color_image(img, lower, upper)
        yellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
        yellow_bin = np.zeros_like(yellow)
        yellow_bin[(yellow >= gray_thresh[0]) & (yellow <= gray_thresh[1])]=1  
            
        # Convert to HLS color space and separate the SL channels
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x filter as in lecture
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel, more yellow
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
        # Create the binary image of all the above channels
        color_binary = np.zeros_like(s_binary)
        color_binary[(sxbinary == 1) | (s_binary == 1) | (yellow_bin == 1)] = 1
    
        return color_binary


    def get_color(self, img):
        
        lower_yellow = np.array([15,0,0])
        upper_yellow = np.array([36, 255, 255])
       
        lower_red = np.array([0,42,42])
        upper_red = np.array([10,255,255])
       
        lower_green = np.array([60,60,60])
        upper_green = np.array([80,255,255])

       
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        img_binary_yellow = cv2.bitwise_and(img,img, mask=yellow_mask)

        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        img_binary_red = cv2.bitwise_and(img,img, mask=red_mask)

        gr_mask = cv2.inRange(hsv, lower_green, upper_green)
        img_binary_gr = cv2.bitwise_and(img,img, mask=gr_mask)
        
        try:
            cv2.imshow("Binary Window R", img_binary_red)
            cv2.imshow("Binary Window Y" , img_binary_yellow)
            cv2.imshow("Binary Window G", img_binary_gr)
            cv2.waitKey(3)

        except CvBridgeError as e:
            print(e)

#         histr = cv2.calcHist([img],[2],None,[256],[0,256]) 
#         found_pos_r = np.argmax(histr[0:histr.shape[0]])
#  
#         histg = cv2.calcHist([img],[1],None,[256],[0,256]) 
#         found_pos_g = np.argmax(histg[0:histg.shape[0]])
#  
#         print('[INFO] R: %d [%d]   G: %d [%d]' % (found_pos_r, 
# 				histr[found_pos_r], 
# 				found_pos_g, 
# 				histg[found_pos_g]))



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

        try:
            cv2.imshow("Detection Window", self.blank_image)
            cv2.waitKey(3)

        except CvBridgeError as e:
            print(e)


        for obj in filtered_results:
            x1_o = obj["bb_o"][0] 
            y1_o = obj["bb_o"][1]
            x2_o = obj["bb_o"][2]
            y2_o = obj["bb_o"][3]
            roi = self.blank_image[y1_o:y2_o, x1_o:x2_o]
            # Use lesson on histograms from CarND-Advanced-Lane-Lines
            self.get_color(roi)
            
        

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
