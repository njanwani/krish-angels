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

class Turn(Enum):
    ROBOT = 0
    PLAYER = 1

# MAKE THIS MESSAGE DEPENDENT EVENTUALLY
TEN = 0
TWENTY = 1
QUEEN = 2
STRIKER = 3
ALL = 4

EE_HEIGHT = 0.181 #0.193
BOARD_HEIGHT = 0.09 #0.07
BOARD_WIDTH = 0.6874
GRIPPER_WIDTH = 0.06
PUCK_RADIUS = 0.015

ZONE_WIDTH = 0.1
ZONE_A_ANG = [np.pi/2, 3*np.pi/2]
ZONE_B_ANG = [np.pi, 2*np.pi]
ZONE_C_ANG = [3*np.pi/2, 5*np.pi/2]
ZONE_D_ANG = [0, np.pi]

endgoalx = 0.05 #round(random.uniform(-0.2, 0.2), 2)
endgoal = np.array([0.5, endgoalx, EE_HEIGHT + BOARD_HEIGHT])
outgoal = np.array([0.14, -0.2, 0])

class Puck:

    def __init__(self, x, angle, denom):
        self.x = x
        self.angle = angle
        self.denom = denom 
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
        self.turn = Turn.ROBOT
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
        
        self.board_center = None
        # check y
        self.zoneA = [-BOARD_WIDTH/2, -BOARD_WIDTH/2 + ZONE_WIDTH]
        self.zoneC = [BOARD_WIDTH/2 - ZONE_WIDTH, BOARD_WIDTH/2]
        # check x
        self.zoneB = [-BOARD_WIDTH/2, -BOARD_WIDTH/2 + ZONE_WIDTH]
        self.zoneD = [BOARD_WIDTH/2 - ZONE_WIDTH, BOARD_WIDTH/2]

        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 1)
        self.armed_sub = self.create_subscription(Bool, '/low_level/armed', self.armed_cb, 1)
        self.board_sub = self.create_subscription(Pose, '/boarddetector//pose', self.board_cb, 1)

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
        self.pucks[ALL] = []
        while self.updating_pucks:
            pass
        self.updating_pucks = True
        for pose in self.puckarray:
            p = Puck(x=np.array([pose.position.x, pose.position.y, EE_HEIGHT + BOARD_HEIGHT]), angle=None, denom=pose.orientation.x)
            self.pucks[pose.orientation.x].append(p)
            self.pucks[ALL].append(p)


        for puck in self.pucks[ALL]:
            if puck.denom != STRIKER:
                continue
            maxr = -np.inf
            minangle = None
            for angle in np.linspace(-np.pi, np.pi, num=25):
                minr = np.inf
                for second in self.pucks[ALL]:
                    if puck == second:
                        continue

                    h = second.x - puck.x                    
                    h = h[:2]
                    hnorm = np.linalg.norm(h)

                    # if hnorm > GRIPPER_WIDTH / 2:
                    #     continueT

                    g = np.array([np.sin(angle), -np.cos(angle)])
                    theta = np.arccos(h @ g / (hnorm * np.linalg.norm(g)))
                    r = hnorm * np.sin(theta)
                    if r < minr:
                        minr = r
                ros_print(self, f'{minr} and {angle}')
                if minr > maxr:
                    # minangle.append(angle)
                    maxr = minr
                    minangle = angle                 
            
            # if minangle == []:
            #     puck.angle = None
            # else:
            #     puck.angle = min(minangle, key=abs)
            puck.angle = minangle
            ros_print(self, f'found angle {puck.angle}')

        self.updating_pucks = False

    def ready_cb(self, msg: Bool):
        self.ready = msg.data
        self.ready_update = True

    def armed_cb(self, msg: Bool):
        self.armed = msg.data
        self.armed_update = True

    def board_cb(self, msg: Pose):
        self.board_center = msg.data
        self.zoneA = [y + self.board_center.y for y in self.zoneA]
        self.zoneC = [y + self.board_center.y for y in self.zoneC]
        self.zoneB = [x + self.board_center.x for x in self.zoneB]
        self.zoneD = [x + self.board_center.x for x in self.zoneD]

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

    def check_angle_bounds(self, puck, angle):
        limit = []
        # find zone
        if puck.x[0] > self.zoneA[0] and puck.x[0] < self.zoneA[1]:
            if puck.y[0] > self.zoneB[0] and puck.y[0] < self.zoneB[1]:
                limit = [max(ZONE_A_ANG[0], ZONE_B_ANG[0]), min(ZONE_A_ANG[1], ZONE_B_ANG[1])]
            elif puck.y[0] > self.zoneD[0] and puck.y[0] < self.zoneD[1]:
                limit = [max(ZONE_A_ANG[0], ZONE_D_ANG[0]), min(ZONE_A_ANG[1], ZONE_D_ANG[1])]
            else:
                limit = ZONE_A_ANG
        
        elif puck.x[0] > self.zoneC[0] and puck.x[0] < self.zoneC[1]:
            if puck.y[0] > self.zoneB[0] and puck.y[0] < self.zoneB[1]:
                limit = [max(ZONE_C_ANG[0], ZONE_B_ANG[0]), min(ZONE_C_ANG[1], ZONE_B_ANG[1])]
            elif puck.y[0] > self.zoneD[0] and puck.y[0] < self.zoneD[1]:
                limit = [max(ZONE_C_ANG[0], ZONE_D_ANG[0]), min(ZONE_C_ANG[1], ZONE_D_ANG[1])]
            else:
                limit = ZONE_C_ANG
        else:
            return True
        
        # check if withstand angle bounds of zones
        return angle <= limit[1] or angle >= limit[0]

    def think(self):
        if self.pucks[STRIKER] == [] and self.puckarray != []:
            # ros_print(self, 'trying to update')
            self.update_pucks()

        if self.pucks[STRIKER] != [] and self.focus is None:
            idx = np.random.choice(np.arange(len(self.pucks[STRIKER])))
            self.focus = self.pucks[STRIKER][idx]
            
        if self.moves == [] and not (self.focus is None):
            # start up
            # robot turn
            # detect/pick up striker, move out of way, pick best puck to shoot, drop puck, move out of way, hit striker
            # player turn
            # update pucks, detect when striker is placed in axis
            if self.turn == Turn.ROBOT:
                self.moves.append(Grab(self.focus.x, angle=self.focus.angle))
                self.moves.append(Move(pos=np.mean(np.vstack((outgoal, endgoal)), axis=0) + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))

                bestpuck = None
                angle = None
                while bestpuck is None:
                    randpuck = random.choice(self.pucks[TEN])
                    temp_angle = np.arctan2(randpuck.x[1]-endgoal[1], randpuck.x[0]-endgoal[0])
                    if self.check_angle_bounds(randpuck, angle):
                        bestpuck = randpuck
                        angle = temp_angle

                self.moves.append(Drop(pos=endgoal)) # change pos to where hit shoot be taken from
                self.moves.append(Move(pos=np.mean(np.vstack((outgoal, endgoal)), axis=0) + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))

                # actual pos of dropped striker
                self.update_pucks()
                striker_pos = self.pucks[STRIKER][0]
                endgoal = np.array([striker_pos.x[0], striker_pos.x[1], EE_HEIGHT + BOARD_HEIGHT + 0.05])
                down = np.array([striker_pos.x[0], striker_pos.x[1], EE_HEIGHT + BOARD_HEIGHT])
                self.moves.append(Move(pos=endgoal, angle=angle))
                self.moves.append(Move(pos=down, angle=angle))

                self.moves.append(Strike(pos=endgoal, angle=angle))
                self.moves.append(Wait(3.0))
                self.moves.append(Move(pos=outgoal + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
                self.turn = Turn.PLAYER

            else:
                # check if striker in pos defined axis and no hands in board (FOR SAFETY)
                self.update_pucks()
                striker_pos = self.pucks[STRIKER][0]
                if striker_pos[1] >= (self.board_center.y - BOARD_WIDTH/2 - 0.085) and striker_pos[1] <= self.board_center.y - BOARD_WIDTH/2 - 0.12:
                    self.turn = Turn.ROBOT
            
            # self.moves.append(Grab(self.focus.x, angle=self.focus.angle))
            # # self.moves.append(Move(pos=outgoal + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
            # # self.update_pucks()
            # # self.moves.append(Wait(3.0))
            # self.moves.append(Move(pos=np.mean(np.vstack((outgoal, endgoal)), axis=0) + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))

            # # change ang to open in a way out of others
            # # self.moves.append(Move(pos=endgoal + np.array([0,0,0.05]), angle=0))
            # self.moves.append(Drop(pos=endgoal))
            # randpuck = random.choice(self.pucks[TEN])
            # angle = np.arctan2(randpuck.x[1]-endgoal[1], randpuck.x[0]-endgoal[0])

            # randpuck = random.choice(self.pucks[TEN])
            # angle = np.arctan2(randpuck.x[1]-endgoal[1], randpuck.x[0]-endgoal[0])
            # self.moves.append(Strike(pos=endgoal, angle=angle))
            # self.moves.append(Wait(3.0))
            # self.moves.append(Move(pos=outgoal + np.array([0,0, EE_HEIGHT + BOARD_HEIGHT + 0.05]), angle=0))
            # # refind striker            

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
