#!/usr/bin/env python3
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ROS Imports
import rclpy
import cv_bridge
from utils.pyutils import *
from enum import Enum
import time


from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool, Float32

RATE = 100.0  
class Mode(Enum):
    START = 0
    TO_PUCK = 1
    GRIP_IT = 2
    TO_HAND = 3
    LET_GO = 4

# MAKE THIS MESSAGE DEPENDENT EVENTUALLY
TEN = 0
TWENTY = 1
QUEEN = 2
STRIKER = 3

EE_HEIGHT = 0.193 #0.141

class Filter:
    def __init__(self, T, x0):
        self.x = x0
        self.T = T
        self.t = time.time()

    def update(self, u):
        u = np.array(u)
        dt = time.time() - self.t
        self.t = time.time()
        self.x = self.x + dt / self.T * (u - self.x)

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
        self.armed = False

        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 10)
        self.armed_sub = self.create_subscription(Bool, '/low_level/armed', self.armed_cb, 10)

        self.goal_pub = self.create_publisher(Pose, '/low_level/goal', 10)
        self.grip_pub = self.create_publisher(Bool, '/low_level/grip', 10)
        self.t = time.time()
        self.t0 = self.t
        self.grip = False

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
        # self.armed = False
        # ros_print(self, 'asdfasdf' + str(self.ready))

    def armed_cb(self, msg: Bool):
        self.armed = msg.data
        # ros_print(self, 'asdfasdf' + str(self.ready))


    def think(self):
        # ros_print(self, self.mode)
        publish = False
        goal = [np.zeros(3), 0]
        grip = False
        self.t = time.time()
        if self.mode == Mode.START:
            pass
        elif self.mode == Mode.TO_PUCK:
            if self.pucks[QUEEN] != []:
                self.to_grab = self.pucks[QUEEN][0]
            
            if self.to_grab == None: return

            goal[0] = np.array([self.to_grab.position.x,
                                self.to_grab.position.y,
                                self.to_grab.position.z + EE_HEIGHT])
            goal[1] = np.arctan2(self.to_grab.position.y, self.to_grab.position.x)
            grip = False
            publish = True
        elif self.mode == Mode.GRIP_IT:
            publish = True
            grip = True
            # FIX GOAL SHIT
        elif self.mode == Mode.TO_HAND:
            goal[0] = np.array([0.5,
                                0.0,
                                EE_HEIGHT + 0.2])
            goal[1] = 0
            grip = True
            publish = True
        elif self.mode == Mode.LET_GO:
            publish = True
            grip = False
        else:
            raise Exception('Invalid mode detected')
        

        if self.mode == Mode.START and self.ready:
            self.mode = Mode.TO_PUCK
            self.t = time.time()
            self.t0 = self.t
        elif self.mode == Mode.TO_PUCK and self.armed:
            self.t = time.time()
            self.t0 = self.t
            self.mode = Mode.GRIP_IT 
        elif self.mode == Mode.GRIP_IT and self.armed and self.t - self.t0 > 0.2:
            self.mode = Mode.TO_HAND
            self.t = time.time()
            self.t0 = self.t
        elif self.mode == Mode.TO_HAND and self.armed and self.t - self.t0 > 0.2:
            self.mode = Mode.LET_GO
        
        if publish:
            posemsg = Pose()
            posemsg.position.x = float(goal[0][0])
            posemsg.position.y = float(goal[0][1])
            posemsg.position.z = float(goal[0][2])
            posemsg.orientation.z = float(goal[1])
            self.goal_pub.publish(posemsg)

            msg = Bool()
            msg.data = grip
            self.grip_pub.publish(msg)

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
