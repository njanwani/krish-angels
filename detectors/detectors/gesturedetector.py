#!/usr/bin/env python3
#
#   mapping.py
#
#   Demostrate how to map pixel coordinates into world coordinates.
#
#   Node:           /mapper
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /mapper/image_raw           Debug (marked up) image
#
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
# ROS Imports
import rclpy
import cv_bridge
from utils.pyutils import *

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool


#
#  Demo Node Class
#
class DemoNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.uc, self.vc = 0, 0
        self.sent = False
        self.recieved = False
        base_options = python.BaseOptions(model_asset_path='/home/robot/robotws/src/detectors/detectors/gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()
        self.successes = 0
        self.last = time.time()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.image = None
        
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
            
        self.successpub = self.create_publisher(Bool, name+'/thumbs',    3)
        self.pubcirc = self.create_publisher(Point, name+'/circle',    3)


    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        if time.time() - self.last <= 0.25:
            return
        self.last = time.time()
        assert(msg.encoding == "rgb8")

        imgRGB = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # STEP 4: Recognize gestures in the input image.
        recognition_result = self.recognizer.recognize(mp_image)

        # STEP 5: Process the result. In this case, visualize it.
        top_gesture = None
        if recognition_result.gestures != []:
            top_gesture = recognition_result.gestures[0][0].category_name
    
        if recognition_result.hand_landmarks:
            for handLms in recognition_result.hand_landmarks:
                for id, lm in enumerate(handLms):
                    h, w, c = imgRGB.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(imgRGB, (cx,cy), 3, (255,0,255), cv2.FILLED)

                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # ros_print(self, top_gesture)
        if top_gesture is not None and top_gesture in ['Thumb_Up', 'Thumb_Down']: self.successes += 1
        else: self.successes = 0

        msg = Bool()
        msg.data = self.successes >= 1
        self.successpub.publish(msg)
        # Convert the image back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(imgRGB, "rgb8"))

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DemoNode('gesture')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()