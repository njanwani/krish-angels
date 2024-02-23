#!/usr/bin/env python3
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ROS Imports
import rclpy
import cv_bridge
from utils.pyutils import *
from enum import Enum


from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool

RATE = 100.0  
class Mode(Enum):
    START = 0
    TO_PUCK = 1
    TO_HAND = 2

# MAKE THIS MESSAGE DEPENDENT EVENTUALLY
TEN = 0
TWENTY = 1
QUEEN = 2
STRIKER = 3

EE_HEIGHT = 0.141

#
#  Detector Node Class
#
class BrainNode(Node):
    # Pick some colors, assuming RGB8 encoding.

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.mode = Mode.START
        self.pucks = {}
        self.pucks[TEN] = []
        self.pucks[TWENTY] = []
        self.pucks[QUEEN] = []
        self.pucks[STRIKER] = []

        self.to_grab = None
        self.ready = False

        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 10)

        self.goal_pub = self.create_publisher(Pose, '/low_level/goal', 10)


        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.think)
        ros_print(self, '\n\n\n\n\n\nNode started\n\n\n\n\n')

    def puck_cb(self, msg: PoseArray):
        
        self.pucks[TEN] = []
        self.pucks[TWENTY] = []
        self.pucks[QUEEN] = []
        self.pucks[STRIKER] = []

        for pose in msg.poses:
            self.pucks[pose.orientation.x].append(pose)

    
    def ready_cb(self, msg: Bool):
        self.ready = msg.data
        # ros_print(self, 'asdfasdf' + str(self.ready))


    def think(self):
        # ros_print(self, self.mode)
        publish = False
        goal = [np.zeros(3), 0]
        if self.mode == Mode.START:
            pass
        elif self.mode == Mode.TO_PUCK:
            if self.pucks[QUEEN] != []:
                self.to_grab = self.pucks[QUEEN][0]
            
            if self.to_grab == None: return

            goal[0] = np.array([self.to_grab.position.x,
                                self.to_grab.position.y,
                                self.to_grab.position.z + EE_HEIGHT])
            goal[1] = 0
            publish = True
        elif self.mode == Mode.TO_HAND:
            pass
        else:
            raise Exception('Invalid mode detected')
        

        if self.mode == Mode.START and self.ready:
            self.mode = Mode.TO_PUCK
        
        
        if publish:
            posemsg = Pose()
            posemsg.position.x = float(goal[0][0])
            posemsg.position.y = float(goal[0][1])
            posemsg.position.z = float(goal[0][2])
            posemsg.orientation.z = 0.0 #float(goal[1])
            self.goal_pub.publish(posemsg)

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
    node = BrainNode('touch')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
