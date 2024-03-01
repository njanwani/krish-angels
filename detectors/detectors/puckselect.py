#!/usr/bin/env python3

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ROS Imports
import rclpy
import cv_bridge
from utils.pyutils import *

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

FPS = 15.0
TEN = 0
TWENTY = 1
QUEEN = 2
STRIKER = 3

#
#  Detector Node Class
#
class DetectorNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red = (255,   0,   0)
    green = (  0, 255,   0)
    blue = (  0,   0, 255)
    yellow = (255, 255,   0)
    white = (255, 255, 255)
    
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.pucks = {}
        self.tree = []
        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()
        self.sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)


    def puck_cb(self, msg: PoseArray):
        self.pucks[TEN] = []
        self.pucks[TWENTY] = []
        self.pucks[QUEEN] = []
        self.pucks[STRIKER] = []

        for pose in msg.poses:
            self.pucks[pose.orientation.x].append(pose)

        # each dude has a tree
        # if min distance to a node is < MAX_DIST
            # add new node to the tree
        # else
            # reset the point in the tree

        



    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('puckdetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
