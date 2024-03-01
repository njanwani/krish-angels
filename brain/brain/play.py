#!/usr/bin/env python3
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ROS Imports
import rclpy
import cv_bridge
from brain_utils.moves import *
from enum import Enum
import time

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import PointStamped, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool, Float32, Header

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

endgoal = np.array([0.5, -0.2, EE_HEIGHT + BOARD_HEIGHT])
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
        self.puckarray = []
        self.updating_pucks = False

        self.focus = None
        self.ready = False
        self.ready_update = 0
        self.armed = False
        self.armed_update = 0
        self.moves = []

        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        # self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 1)
        # self.armed_sub = self.create_subscription(Bool, '/low_level/armed', self.armed_cb, 1)
        self.goal_received_sub = self.create_subscription(Header, 'low_level/goal_received', self.goal_received, 1)
        self.ready_armed_sub = self.create_subscription(PointStamped, 'low_level/ready_armed', self.ready_armed_cb, 1)
        self.ready_armed_update_t = 0
        self.last_move_t = -1

        self.goal_pub = self.create_publisher(Pose, '/low_level/goal_2', 10)
        self.grip_pub = self.create_publisher(Bool, '/low_level/grip', 10)
        self.strike_pub = self.create_publisher(Float32, '/end_effector/strike', 10)
        self.t = time.time()
        self.t0 = self.t
        self.grip = False

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.think)
        rate       = 20.0
        self.timer = self.create_timer(1 / rate, self.work)

    def goal_received(self, msg: Header):
        ros_print(self, 'I AM RECEIVED')
        self.last_move_t = float(msg.stamp.sec)

    def puck_cb(self, msg: PoseArray):
        while self.updating_pucks:
            pass

        self.updating_pucks = True
        self.puckarray = msg.poses
        # ros_print(self, msg.poses)
        self.updating_pucks = False

    def update_pucks(self):
        self.pucks[TEN] = []
        self.pucks[TWENTY] = []
        self.pucks[QUEEN] = []
        self.pucks[STRIKER] = []
        while self.updating_pucks:
            pass
        self.updating_pucks = True
        for pose in self.puckarray:
            self.pucks[pose.orientation.x].append(np.array([pose.position.x, pose.position.y, EE_HEIGHT + BOARD_HEIGHT]))
        self.updating_pucks = False

    def ready_armed_cb(self, msg: PointStamped):
        self.ready_armed_update_t = float(msg.header.stamp.sec)
        self.ready = bool(float(msg.point.x))
        self.armed = bool(float(msg.point.y))
        ros_print(self, f'aklsjd {self.armed}')

    def work(self):
        self.t, _ = self.get_clock().now().seconds_nanoseconds()
        if self.moves == []:
            return
        
        if self.last_move_t >= self.ready_armed_update_t:
            return
        
        # ros_print(self, f'{type(self.moves[0])} and {self.armed} and {self.ready} and {self.moves[0].mode}')
        goal, grip, strike = self.moves[0].step(self.t, self.ready, self.armed)

        if not (goal is None):
            # ros_print(self, f'{self.moves[0].mode}')
            posemsg = Pose()
            posemsg.position.x = float(goal[0][0])
            posemsg.position.y = float(goal[0][1])
            posemsg.position.z = float(goal[0][2])
            posemsg.orientation.z = float(goal[1])
            self.goal_pub.publish(posemsg)
            # self.last_move_t, _ = self.get_clock().now().seconds_nanoseconds()

        if not (grip is None):
            msg = Bool()
            msg.data = grip
            self.grip_pub.publish(msg)
            # self.last_move_t, _ = self.get_clock().now().seconds_nanoseconds()

        if not (strike is None):
            msg = Float32()
            msg.data = float(strike)
            ros_print(self, f'sending {strike}')
            self.strike_pub.publish(msg)
            # self.last_move_t, _ = self.get_clock().now().seconds_nanoseconds()

        if self.moves[0].done:
            ros_print(self, 'MOVING ON')
            self.moves.pop(0)


    def think(self):
        # ros_print(self, self.mode)
        if self.pucks[TEN] == [] and self.puckarray != []:
            ros_print(self, 'trying to update')
            self.update_pucks()

        if self.pucks[TEN] != [] and self.focus is None:
            idx = np.random.choice(np.arange(len(self.pucks[TEN])))
            self.focus = self.pucks[TEN][idx]
            
        if self.moves == [] and not (self.focus is None):
            self.moves.append(Grab(self.focus))
            # self.moves.append(Wait(3.0))
            self.moves.append(Move(pos=np.mean(np.vstack((self.focus, endgoal)), axis=0) + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
            # self.moves.append(Wait(3.0))
            self.moves.append(Drop(pos=endgoal))
            # self.moves.append(Wait(3.0))
            self.moves.append(Strike(pos=endgoal, angle=0))
            self.moves.append(Wait(3.0))


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
    node = BrainNode('play')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
