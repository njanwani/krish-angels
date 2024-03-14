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

RATE = 1.0  
class Stage(Enum):
    GET = 1
    PUT = 2
    SHOOT = 3
    RETURN = 4

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
BOARD_HEIGHT = 0.075 #0.07
REST_HEIGHT = np.array([0,0,0.1])
BOARD_WIDTH = 0.6874
GRIPPER_WIDTH = 0.06
PUCK_RADIUS = 0.015

dist = {(0,1) : np.pi,
        (1,3) : 3 * np.pi / 2,
        (3,2) : 0,
        (2,0) : np.pi / 2}

ZONE_WIDTH = 0.1
ZONE_ANGS = {}
ZONE_ANGS['A'] = [3 * np.pi / 2, 5 * np.pi / 2]
ZONE_ANGS['B'] = [np.pi, 2 * np.pi]
ZONE_ANGS['C'] = [np.pi / 2, 3 * np.pi / 2]
ZONE_ANGS['D'] = [0, np.pi]

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
    HOME = np.array([0.0, 0.52, 0.5])
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.num_pucks = 20
        self.stage = Stage.GET
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
        self.board_corners = None
        self.shot_axis = None
        # # check y
        # self.zoneA = lambda center: ([center.x - BOARD_WIDTH / 2 + ZONE_WIDTH, center.y - BOARD_WIDTH / 2 + ZONE_WIDTH])
        # self.zoneC = [BOARD_WIDTH/2 - ZONE_WIDTH, BOARD_WIDTH/2]
        # # check x
        # self.zoneB = [-BOARD_WIDTH/2, -BOARD_WIDTH/2 + ZONE_WIDTH]
        # self.zoneD = [BOARD_WIDTH/2 - ZONE_WIDTH, BOARD_WIDTH/2]
        self.thumbs_up = False
        self.puck_sub = self.create_subscription(PoseArray, '/puckdetector/pucks', self.puck_cb, 10)
        self.ready_sub = self.create_subscription(Bool, '/low_level/ready', self.ready_cb, 1)
        self.armed_sub = self.create_subscription(Bool, '/low_level/armed', self.armed_cb, 1)
        self.board_sub = self.create_subscription(Pose, '/boarddetector/pose', self.board_cb, 1)
        self.board_corners_sub = self.create_subscription(PoseArray, '/boarddetector/board_corners', self.board_corners_cb, 1)
        self.shot_axis_sub = self.create_subscription(PoseArray, '/boarddetector/shot_axis', self.shot_axis_cb, 1)
        self.thumbs_up_sub = self.create_subscription(Bool, '/gesture/thumbs', self.thumbs_up_cb, 1)

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

    def in_board(self, puck):
        if self.board_corners is None:
            return False
        xmin, ymin = self.board_corners[1].position.x, self.board_corners[1].position.y
        xmax, ymax = self.board_corners[2].position.x, self.board_corners[2].position.y
        return xmin <= puck.x[0] and puck.x[0] <= xmax and ymin <= puck.x[1] and puck.x[1] <= ymax
        
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
            if self.in_board(p):
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
                # ros_print(self, f'{minr} and {angle}')
                if minr > maxr:
                    # minangle.append(angle)
                    maxr = minr
                    minangle = angle
            # if minangle == []:
            #     puck.angle = None
            # else:
            #     puck.angle = min(minangle, key=abs)
            if minangle < 0:
                if abs(minangle + np.pi) < abs(minangle):
                    minangle = minangle + np.pi
            else:
                if abs(minangle - np.pi) < abs(minangle):
                    minangle = minangle - np.pi

            sides = [(0,1),
                     (1,3),
                     (3,2),
                     (2,0)]
           
                
            for pose1, pose2 in sides:
                if self.board_corners is None:
                    break
                side = pose1, pose2
                pose1 = self.board_corners[pose1]
                pose2 = self.board_corners[pose2]
                c1 = np.array([pose1.position.x, pose1.position.y])
                c2 = np.array([pose2.position.x, pose2.position.y])
                p = np.array([puck.x[0], puck.x[1]]) - c1
                s = c2 - c1
                theta = np.arccos(s @ p / (np.linalg.norm(p) * np.linalg.norm(s)))
                d = np.abs(np.linalg.norm(p) * np.sin(theta))
                ros_print(self, f'{side}: {d}')
                if d < 0.1:
                    minangle = dist[side]
                    break

            puck.angle = minangle
            # ros_print(self, f'found angle {puck.angle}')

        self.updating_pucks = False

    def pick_puck(self):
        dmin, puckmin = np.inf, None
        for puck in self.pucks[ALL]:
            if puck.denom == STRIKER:
                continue
            ros_print(self, 'trying a puck')
            temp_dmin = np.inf
            for corner in self.board_corners:
                d = np.array([puck.x[0], puck.x[1]]) - np.array([corner.position.x, corner.position.y])
                if np.linalg.norm(d) < temp_dmin:
                    temp_dmin = np.linalg.norm(d)

            if temp_dmin < dmin:
                dmin = temp_dmin
                puckmin = puck
        
        return puckmin

    def pick_shot_pos(self):
        positions = np.linspace([self.shot_axis[0].position.x, self.shot_axis[0].position.y, BOARD_HEIGHT + EE_HEIGHT],
                                [self.shot_axis[1].position.x, self.shot_axis[1].position.y, BOARD_HEIGHT + EE_HEIGHT],
                                num=20)
        ret = []
        for pos in positions:
            pos_ok = True
            for puck in self.pucks[ALL]:
                if puck.denom == STRIKER:
                    continue
                
                p = puck.x[:2]
                if np.linalg.norm(pos[:2] - p) < 0.07:
                    pos_ok = False
                    break
            
            if pos_ok:
                ret.append(pos)
        
        return np.array(ret)

    def ready_cb(self, msg: Bool):
        self.ready = msg.data
        self.ready_update = True

    def armed_cb(self, msg: Bool):
        self.armed = msg.data
        self.armed_update = True

    def board_cb(self, msg: Pose):
        self.board_center = msg.position

    def board_corners_cb(self, msg: PoseArray):
        self.board_corners = msg.poses

    def thumbs_up_cb(self, msg: Bool):
        self.thumbs_up = msg.data

    def shot_axis_cb(self, msg: PoseArray):
        # ros_print(self, 'shot_axis_cb')
        self.shot_axis = msg.poses

    def transit_stage(self):
        if self.stage == Stage.GET: self.stage = Stage.PUT
        elif self.stage == Stage.PUT: self.stage = Stage.SHOOT
        elif self.stage == Stage.SHOOT: self.stage = Stage.RETURN
        else: raise Exception('Unknown stage encountered')

    def work(self):
        self.t = time.time()
        if self.moves == []:
            return
        
        # if not self.armed_update or not self.ready_update:
        #     # ros_print(self, 'LATE UPDATE')
        #     time.sleep(0.2)
        #     return
        # ros_print(self, f'{type(self.moves[0])} and {self.armed} and {self.ready} and {self.moves[0].mode}')
        goal, grip, strike = self.moves[0].step(self.t, self.ready, self.armed)
        self.armed_update = False
        self.ready_update = False

        if not (goal is None):
            posemsg = Pose()
            posemsg.position.x = float(goal[0][0])
            posemsg.position.y = float(goal[0][1])
            posemsg.position.z = float(goal[0][2])
            posemsg.orientation.z = float(goal[1])
            posemsg.orientation.y = float(goal[2])
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
                self.transit_stage()
                self.update_pucks()

    def think(self):
        # ros_print(self, self.stage)
        if self.turn == Turn.ROBOT:
            if self.stage == Stage.GET and self.moves == []:
                self.update_pucks()

                if self.pucks[STRIKER] != [] and self.focus is None:
                    idx = np.random.choice(np.arange(len(self.pucks[STRIKER])))
                    self.focus = self.pucks[STRIKER][idx]

                    self.moves.append(Move(pos=self.focus.x + REST_HEIGHT, angle=self.focus.angle))
                    self.moves.append(Grab(self.focus.x, angle=self.focus.angle))
                    self.moves.append(Move(pos=self.focus.x + REST_HEIGHT, angle=self.focus.angle))
                    self.moves.append(Move(pos=self.HOME, angle=0))

            elif self.stage == Stage.PUT and self.moves == []:
                # CURRENTLY RANDOMIZED SHOT, REPLACE WITH A PLAY ALGORITHM
                self.update_pucks()
                self.shoot_pos, self.shoot_angle = np.array([0.4, 0.0, BOARD_HEIGHT + EE_HEIGHT]), 0.0 # SWITCH WITH PLAY ALGO
                if self.shot_axis is not None:
                    positions = self.pick_shot_pos()
                    self.shoot_pos = positions[np.random.choice(list(range(0, len(positions))))]
                    # ros_print(self, self.shot_axis)
                    # ros_print(self, self.shoot_pos + np.array([0, 0, 0.2]))

                pucks = self.pucks[TEN] + self.pucks[TWENTY] + self.pucks[QUEEN]
                ros_print(self, len(pucks))
                if pucks is not None or pucks is not []:
                    # randpuck = random.choice(pucks)
                    randpuck = self.pick_puck()
                    self.shoot_angle = -np.arctan2(randpuck.x[1]-self.shoot_pos[1], randpuck.x[0]-self.shoot_pos[0])
                ros_print(self, self.shoot_angle)
                self.moves.append(Move(pos=self.shoot_pos + REST_HEIGHT, angle=self.shoot_angle))
                self.moves.append(Drop(self.shoot_pos, angle=self.shoot_angle))
                self.moves.append(Move(pos=self.shoot_pos + REST_HEIGHT, angle=self.shoot_angle))
                self.moves.append(Move(pos=self.HOME, angle=0))
                
            elif self.stage == Stage.SHOOT and self.moves == []:
                self.update_pucks()
                if self.pucks[STRIKER] != [] and self.focus is None:
                    idx = np.random.choice(np.arange(len(self.pucks[STRIKER])))
                    self.focus = self.pucks[STRIKER][idx]

                    pucks = self.pucks[TEN] + self.pucks[TWENTY] + self.pucks[QUEEN]
                    if pucks is not None or pucks is not []:
                        randpuck = random.choice(pucks)
                        self.shoot_angle = np.arctan2(randpuck.x[1]-self.focus.x[1], randpuck.x[0]-self.focus.x[0])

                    delta = 0.02 * np.array([np.cos(self.shoot_angle + np.pi), np.sin(self.shoot_angle + np.pi), 0])
                    ros_print(self, "SHOOT DEBUG: ")
                    ros_print(self, "SHOOT ANGLE: " + str(self.shoot_angle))
                    ros_print(self, "DELTA: " + str(delta) + "\n\n")
                    self.moves.append(Move(pos=self.focus.x + delta + REST_HEIGHT, angle=self.shoot_angle))
                    self.moves.append(Strike(self.focus.x + delta - np.array([0, 0, 0.03]), angle=self.shoot_angle))
                    ros_print(self, "DEBUG MESSAGE: adding a strike command")
                    self.moves.append(Wait(3.0))
                    self.moves.append(Move(pos=self.focus.x + delta + REST_HEIGHT, angle=self.shoot_angle))
                    self.moves.append(Move(pos=self.HOME, angle=0))

            elif self.stage == Stage.RETURN:
                self.turn = Turn.PLAYER
            elif self.stage not in Stage:
                raise Exception(f'Unknown Stage encountered: {self.stage}')

        else:
            # check if striker in pos defined axis and no hands in board (FOR SAFETY)
            self.update_pucks()
            if self.thumbs_up:
                self.stage = Stage.GET
                self.turn = Turn.ROBOT

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
