#!/usr/bin/env python3
import cv2
import numpy as np
from sklearn.cluster import KMeans
import random

# ROS Imports
import rclpy
import cv_bridge
from brain_utils.moves import *
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

EE_HEIGHT = 0.191 #0.193
BOARD_HEIGHT = 0.07

endgoalx = 0.05 #round(random.uniform(-0.2, 0.2), 2)
endgoal = np.array([0.5, endgoalx, EE_HEIGHT + BOARD_HEIGHT])
outgoal = np.array([0.14, -0.2, 0])
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
        self.ready_update = False
        self.armed = False
        self.armed_update = False
        self.moves = []

        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 1)
        self.armed_sub = self.create_subscription(Bool, '/low_level/armed', self.armed_cb, 1)

        self.goal_pub = self.create_publisher(Pose, '/low_level/goal_2', 10)
        self.grip_pub = self.create_publisher(Bool, '/low_level/grip', 10)
        self.strike_pub = self.create_publisher(Float32, '/end_effector/strike', 10)
        self.t = time.time()
        self.t0 = self.t
        self.grip = False

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.think)
        rate       = 10.0
        self.timer = self.create_timer(1 / rate, self.work)
        ros_print(self, 'STARTED PLAY')

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

    def ready_cb(self, msg: Bool):
        self.ready = msg.data
        self.ready_update = True

    def armed_cb(self, msg: Bool):
        self.armed = msg.data
        self.armed_update = True

    def work(self):
        self.t = time.time()
        if self.moves == []:
            return
        
        if not self.armed_update or not self.ready_update:
            # ros_print(self, 'LATE UPDATE')
            time.sleep(0.2)
            return
        # ros_print(self, f'{type(self.moves[0])} and {self.armed} and {self.ready} and {self.moves[0].mode}')
        goal, grip, strike = self.moves[0].step(self.t, self.ready, self.armed)
        self.armed_update = False
        self.ready_update = False

        if not (goal is None):
            # ros_print(self, f'{self.moves[0].mode}')
            posemsg = Pose()
            posemsg.position.x = float(goal[0][0])
            posemsg.position.y = float(goal[0][1])
            posemsg.position.z = float(goal[0][2])
            posemsg.orientation.z = float(goal[1])
            self.goal_pub.publish(posemsg)
            self.armed = False
            self.ready = False

        if not (grip is None):
            msg = Bool()
            msg.data = grip
            self.grip_pub.publish(msg)

        if not (strike is None):
            msg = Float32()
            msg.data = float(strike)
            ros_print(self, f'sending {strike}')
            self.strike_pub.publish(msg)

        if self.moves[0].done:
            ros_print(self, 'MOVING ON')
            self.moves.pop(0)
            if self.moves == []:
                self.focus = None
                self.update_pucks()


    def think(self):
        # ros_print(self, self.mode)
        if self.pucks[STRIKER] == [] and self.puckarray != []:
            # ros_print(self, 'trying to update')
            self.update_pucks()

        if self.pucks[STRIKER] != [] and self.focus is None:
            idx = np.random.choice(np.arange(len(self.pucks[STRIKER])))
            self.focus = self.pucks[STRIKER][idx]
            
        if self.moves == [] and not (self.focus is None):
            # grab striker
            # move out of the way to see all possible pucks to hit
            # update pucks
            # choose best puck shoot at 
            # move to viable user player area, choose angle relative to the chosen puck
            # drop striker to that pos
            # strike
            # wait
            
            self.moves.append(Grab(self.focus))
            # self.moves.append(Move(pos=outgoal + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
            # self.update_pucks()
            # self.moves.append(Wait(3.0))
            self.moves.append(Move(pos=np.mean(np.vstack((outgoal, endgoal)), axis=0) + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))

            # change ang to open in a way out of others
            # self.moves.append(Move(pos=endgoal + np.array([0,0,0.05]), angle=0))
            self.moves.append(Drop(pos=endgoal, droppos=endgoal + np.array([0.0, 0.0, 0.1 + EE_HEIGHT + BOARD_HEIGHT])))
            randpuck = random.choice(self.pucks[TEN])
            angle = -np.arctan2(randpuck[1]-endgoal[1], randpuck[0]-endgoal[0])
            self.moves.append(Strike(pos=endgoal, angle=angle))
            ros_print(self, f'angle dude {angle}')
            self.moves.append(Wait(3.0))
            self.moves.append(Move(pos=outgoal + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
            # refind striker            

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
