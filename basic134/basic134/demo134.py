#!/usr/bin/env python3

import rclpy
from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float32

from utils.KinematicChain import KinematicChain as KC
from utils.TrajectoryUtils import *
from utils.pyutils import *

from enum import Enum
import numpy as np
import time

RATE = 100.0
RECT_WIDTH = 26 * 10**(-3) * 0.5 #26
COMMAND_TOPIC = '/joint_commands'
FEEDBACK_TOPIC = '/joint_states'
CIRC_TOPIC = '/balldetector/circle'
RECT_TOPIC = 'balldetector/rectangle'

class Mode(Enum):
    JOINT_SPLINE      = 0
    TASK_SPLINE       = 1
    HOLD              = 2
    CONTACTED         = 3

class DemoNode(Node):

    VMAX    = 0.8
    AMAX    = VMAX / 10
    THOLD   = 1

    def __init__(self, name):
        super().__init__(name)

        self.JS = {}
        self.TS = {}

        # Initialization
        self.chain = KC(self, 'world', 'tip', ['base', 'shoulder', 'elbow'])
        self.position0      = np.array(self.grabfbk())
        self.contacted = False

        # joint space
        self.curr_pos = np.array(self.position0) 
        self.curr_vel = np.array([0, 0, 0])
        self.curr_eff = np.array([0, 0, 0])
        self.joint_wait = np.array([0, np.pi/2, -np.pi/2])
        self.qdot_filter = self.curr_vel.copy()
        self.qdot_T = 0.5
        self.q0 = self.position0

        # task space
        self.position_init = self.chain.fkin(self.position0)[0].flatten()
        self.mag_zeropos = np.linalg.norm(self.chain.fkin(np.zeros(3))[0].flatten())
        self.position_wait = self.chain.fkin(self.joint_wait)[0].flatten() #np.array([0.29972, -0.0508, 0.50607])

        # Setup the queue
        self.queue = []
        self.qlast = self.curr_pos.reshape((3,1))

        # setup beginning motion in JS
        self.mode = Mode.JOINT_SPLINE
        self.t = 0 #time.time()
        self.t0 = self.t
        self.tmove = splinetime(self.curr_pos,      # p0    
                                self.joint_wait,    # pf
                                np.zeros(3),        # v0
                                np.zeros(3),        # vf
                                cartesian=False)

        # Detection gains for contact detection
        # TODO: TUNE THESE HOES
        self.pG = 0    # position
        self.vG = 0 #1.5   # velocity
        self.eG = 1.0     # effort
        self.thresh_contact = 1

        # Gravity compenstation gains
        self.B = 1.5
        self.C = 0.1

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, COMMAND_TOPIC, 10)

        self.err_msg = Float32()
        self.err_pub = self.create_publisher(Float32, '/pos_error', 10)

        # Create a publisher for printing position error
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, COMMAND_TOPIC, 10)

        # Wait for a connection to happen.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers(COMMAND_TOPIC)):
            pass

        # Create a subscriber to receive point messages.
        self.circsub = self.create_subscription(Point, CIRC_TOPIC, self.recvcirc, 10)
        
        # Create a subscriber to receive point messages.
        self.rectsub = self.create_subscription(Pose, RECT_TOPIC, self.recvrect, 10)

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(JointState, FEEDBACK_TOPIC, self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
    
    def gravity(self, pos):
        humerus_component = float(self.B * np.cos(pos[1]))
        forearm_component = float(self.C * np.cos(pos[1] + pos[2])) #* np.cos(pos[2])))
        return np.array([0.0, humerus_component + forearm_component, 0.0])
    
    def shutdown(self):
        self.destroy_node()

    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        return self.grabpos
    
    def recvfbk(self, fbkmsg):
        # Receive feedback - called repeatedly by incoming messages.
        self.curr_pos = fbkmsg.position
        self.curr_vel = fbkmsg.velocity
        self.curr_eff = fbkmsg.effort
        self.qdot_filter = self.qdot_filter + (0.01 / self.qdot_T * (self.curr_vel - self.qdot_filter))
        
    # rejkavik, iceland a point
    def recvcirc(self, pointmsg):
        # Receive a point message - called by incoming messages.
        x, y, z = pointmsg.x, pointmsg.y, pointmsg.z
        self.queue.append(bound_taskspace(np.array([x,y,z]), self.mag_zeropos))
    
    def recvrect(self, msg: Pose):
        # Receive a point message - called by incoming messages.
        x, y, z = msg.position.x, msg.position.y, msg.position.z
        theta = msg.orientation.z
        p1 = RECT_WIDTH * np.array([np.sin(theta), np.cos(theta), 0]) + np.array([x,y,z])
        p2 = np.array([x,y,z + 0.03])
        p3 = RECT_WIDTH * np.array([np.sin(theta + np.pi), np.cos(theta + np.pi), 0]) + np.array([x,y,z])
        self.queue.append(bound_taskspace(p1, self.mag_zeropos))
        self.queue.append(bound_taskspace(p2, self.mag_zeropos))
        self.queue.append(bound_taskspace(p3, self.mag_zeropos))

    def is_contacted(self, cmdpos, cmdvel, cmdeff):
        error_eff = sum(np.abs(self.curr_eff - cmdeff))
        error_vel = sum(np.abs(self.curr_vel - cmdvel))
        error_pos = sum(np.abs(self.curr_pos - cmdpos))
        contacted = self.pG * error_pos + self.vG * error_vel + self.eG * error_eff > self.thresh_contact
        # ros_print(self, str(cmdpos - self.curr_pos))
        
        if contacted:
            self.queue = []

        return contacted

    def is_stopped(self, thresh = 0.01):
        y = (np.linalg.norm(self.qdot_filter) < thresh)
        self.err_msg.data = np.linalg.norm(self.qdot_filter)
        self.err_pub.publish(self.err_msg)
        return y
    
    def compute_ts_spline(self):
        p_last, _ = spline5(self.t - self.t0,
                            self.tmove,
                            self.chain.fkin(self.q0)[0].flatten(),
                            self.queue[0],
                            0, 0, 0, 0)
        _, v = spline5(self.t + 1.0 / RATE - self.t0,
                       self.tmove,
                       self.chain.fkin(self.q0)[0].flatten(),
                       self.queue[0],
                       0, 0, 0, 0)
        
        q, qdot = self.chain.ikin(1.0 / RATE, self.qlast.reshape((3,1)), p_last.reshape((3,1)), v.reshape((3,1)))
        return q, qdot
    
    def check_switch_modes(self, q):
        if self.contacted:
            self.mode = Mode.CONTACTED
            self.t0 = self.t
            self.tmove = np.inf
            if self.is_stopped():
                ros_print(self, 'Stopped!')
                self.mode = Mode.JOINT_SPLINE
                self.t0 = self.t
                self.q0 = self.curr_pos
                self.tmove = splinetime(self.q0,            # p0    
                                        self.joint_wait,    # pf
                                        np.zeros(3),        # v0
                                        np.zeros(3),        # vf
                                        cartesian=False)
                self.contacted = False
                
            
        elif self.mode == Mode.TASK_SPLINE and self.t - self.t0 > self.tmove:
            self.queue.pop(0)
            self.mode = Mode.HOLD
            self.t0 = self.t
            self.tmove = DemoNode.THOLD
        elif self.mode == Mode.HOLD and len(self.queue) > 0:
            self.mode = Mode.TASK_SPLINE
            self.t0 = self.t
            self.tmove = splinetime(self.chain.fkin(q)[0],
                                    self.queue[0],
                                    np.zeros(3),
                                    np.zeros(3))
            self.q0 = q
        elif self.mode == Mode.JOINT_SPLINE and self.t - self.t0 > self.tmove:
            self.mode = Mode.HOLD
            self.t0 = self.t
            self.tmove = DemoNode.THOLD
            self.q0 = q
    
    def sendcmd(self):
        # Send a command - called repeatedly by the timer.
        self.t += 0.01
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow']
        if self.mode == Mode.TASK_SPLINE:
            q, qdot = self.compute_ts_spline()
            qeff = self.gravity(q)

        elif self.mode == Mode.CONTACTED:
            q = np.array([self.qlast[0], np.nan, np.nan])
            qdot = np.array([np.nan, np.nan, np.nan])
            qeff = self.gravity(self.curr_pos)

        elif self.mode == Mode.HOLD:
            q, qdot = self.qlast, np.zeros(3)
            qeff = self.gravity(q)

        elif self.mode == Mode.JOINT_SPLINE:
            q, qdot = spline5(self.t - self.t0, self.tmove, self.q0, self.joint_wait, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
            qeff = self.gravity(q)

        else:
            ros_print(self, "I fucked up bad...")
            raise Exception("Zamn")

        self.qlast = q
        self.contacted = self.is_contacted(q, qdot, qeff) or self.contacted
        self.check_switch_modes(q)
        
        
        self.cmdmsg.position =  q.flatten().tolist()
        self.cmdmsg.velocity =  qdot.flatten().tolist()
        self.cmdmsg.effort =    qeff.flatten().tolist()
        self.cmdpub.publish(self.cmdmsg)


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('demo')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
