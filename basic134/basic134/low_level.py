#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import rclpy
import time
from utils.KinematicChain import KinematicChain as KC
from utils.TrajectoryUtils import *
from utils.TransformHelpers import *
from enum import Enum
from utils.pyutils import *

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float32, Bool

#
#   Definitions
#
RATE = 100.0            # Hertz


class Mode(Enum):
    JOINT = 1
    TASK = 2
    GRAV = 3
    START = 4
    STILL = 5

class RobotPose:
    def __init__(self, x, R):
        self.x = x
        self.R = R

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
#   DEMO Node Class
#
class DemoNode(Node):

    VMAX    = 0.8
    AMAX    = VMAX / 10
    THOLD   = 1

    GRIP_MIN = -1
    GRIP_MAX = 1

    POINT_LIB = (np.array([0.45, 0.15, 0.0]),           # side x
                 np.array([0.30, 0.0, 0.0]),            # middle x
                 np.array([0, 0, 50.0]),                # out of task space + singularity
                 np.array([1.5, .1725, 0.01]),          # out of task space
    )

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.chain = KC(self, 'world', 'endmotor', ['base', 'shoulder', 'elbow', 'wrist', 'end'])
        self.position0 = self.grabfbk()[:-1]
        self.queue = []

        self.t = 0
        self.t0 = self.t
        self.tmove = 2
        self.mode = Mode.GRAV

        self.JS = {}
        self.JS['q0'] = np.array(self.position0)
        self.JS['q_act'] = None
        self.JS['q_last'] = self.JS['q0']
        self.JS['q_start'] = np.array([0.0, -np.pi / 2, -np.pi / 2, 0.0, 0.0])

        self.TS = {}
        self.TS['p0'] = RobotPose(*(self.chain.fkin(self.position0)[:2]))
        self.TS['goal'] = RobotPose(*(self.chain.fkin(self.JS['q_start'])[:2]))
        self.TS['v0'] = 0
        self.TS['a0'] = 0

        self.s_last = 0
        self.sdot_last = 0
        self.sdotdot_last = 0

        self.effort = Filter(1, 0)
        self.grip = 0

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(JointState, '/joint_states', self.cb_states, 10)
        self.grip_sub = self.create_subscription(Float32, '/grip', self.cb_grip, 10)
        
        ros_print(self, 'waiting for feedback')
        while self.JS['q_act'] is None:
            rclpy.spin_once(self)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        self.ready = self.create_publisher(Bool, '/' + name + '/ready', 10)

        while not self.count_subscribers('/joint_commands'):
            pass

        # Create a subscriber to receive point messages.
        self.recvpt_sub = self.create_subscription(Pose, f'/{name}/goal', self.recvpt, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.sendcmd)
        ros_print(self, 'Node started')
    
    def cb_grip(self, msg: Float32):
        self.grip = min(max(msg.data, self.GRIP_MIN), self.GRIP_MAX)

    def state_space(self):
        if self.mode == Mode.STILL:
            return 0,0,0
        
        return spline5(self.t - self.t0, self.tmove, p0=self.s_last,
                                                     pf=1,
                                                     v0=self.sdot_last,
                                                     vf=0,
                                                     a0=self.sdotdot_last,
                                                     af=0)

    def recvpt(self, msg: Pose):
        if np.all(self.TS['goal'].x == self.TS['p0'].x) and np.all(self.TS['goal'].R == self.TS['p0'].R):
            return 
        
        R = Reye() @ Rotx(0) @ Roty(0) @ Rotz(msg.orientation.z)

        if np.all(np.isclose(np.array([msg.position.x, msg.position.y, msg.position.z]), self.TS['goal'].x, atol=0.01)) and np.all(np.isclose(R, self.TS['goal'].R, atol=0.01)):
            return
        
        v_last, a_last = 0,0
        if self.mode == Mode.TASK:
            _, v_last, a_last = spline5(self.t - self.t0, self.tmove, self.TS['p0'].x.flatten(), self.TS['goal'].x.flatten(), self.TS['v0'],0, self.TS['a0'], 0)
            v_last, a_last = v_last.flatten(), a_last.flatten()
            
        self.TS['goal'].x = np.array([msg.position.x,
                                      msg.position.y,
                                      msg.position.z])
        self.TS['goal'].R = R
        self.TS['p0'] = RobotPose(*(self.chain.fkin(self.JS['q_last'])[:2]))


        if self.mode in [Mode.STILL, Mode.TASK]:
            self.mode = Mode.TASK
            self.t0 = self.t
            self.tmove = splinetime(self.TS['p0'].x, self.TS['goal'].x, v0=v_last, vf=0)
            self.s_last = 0
            self.TS['v0'] = v_last
            self.TS['a0'] = a_last
            ros_print(self, f'vlast = {v_last}')
    
    # Save the actual position.
    def cb_states(self, msg: JointState):
        self.JS['q_act'] = np.array(msg.position)[:-1]
        if self.mode == Mode.START:
            self.JS['q0'] = self.JS['q_act']
        self.effort.update(msg.effort[:-1])
        ros_print(self, self.effort)
    
    def gravity(self):
        t_elbow = 8 * np.cos(self.JS['q_act'][1] - self.JS['q_act'][2])
        t_shoulder = -t_elbow - 9.8 * np.cos(self.JS['q_act'][1])
        return t_shoulder, t_elbow

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()

    # Grab a single feedback - do not call this repeatedly.
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

        # Return the values.
        return self.grabpos
    

    def compute_ts_spline(self, s, sdot):
        p, v, _ = spline5(self.t - self.t0, self.tmove, self.TS['p0'].x.flatten(), self.TS['goal'].x.flatten(), self.TS['v0'],0, self.TS['a0'], 0)
        R = Rinter(self.TS['p0'].R, self.TS['goal'].R, s)
        w = winter(self.TS['p0'].R, self.TS['goal'].R, sdot)

        # ros_print(self, f'p {p}')
        
        q, qdot = self.chain.ikin(1 / RATE, self.JS['q_act'], p.reshape((3,1)), v.reshape((3,1)), w, R)
        return q, qdot

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        self.t += 1 / RATE
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'end', 'grip']
        nan = float("nan")
        ready = False
        
        t_shoulder, t_elbow = self.gravity()
        q, qdot, qeff = None, None, None
        if self.mode == Mode.START:
            q = [nan] * 5
            qdot = [nan] * 5
            qeff = spline5(self.t - self.t0, self.tmove, 0, 1, 0, 0, 0, 0)[0] * np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])

        elif self.mode == Mode.GRAV:
            q = [nan] * 5
            q[1] = -np.pi / 2
            qdot = [nan] * 5
            qeff = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])

        elif self.mode == Mode.TASK:
            s, sdot, _ = self.state_space()
            q, qdot = self.compute_ts_spline(s, sdot)
            qeff = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])
            ready = True

        elif self.mode == Mode.JOINT:
            q, qdot, _ = spline5(self.t - self.t0, self.tmove, self.JS['q0'], self.JS['q_start'], 0, 0, 0, 0)
            qeff = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])

        elif self.mode == Mode.STILL:
            q, qdot = self.JS['q_last'], np.zeros(5)
            qeff = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])
            ready = True
        
        else:
            raise Exception('OOPS')
        
        self.JS['q_last'] = q
        self.JS['qd_last'] = qdot

        if self.mode == Mode.START and self.t - self.t0 > self.tmove:
            self.mode = Mode.JOINT
            self.tmove = splinetime(self.JS['q0'], self.JS['q_start'], 0, 0, DemoNode.VMAX, DemoNode.AMAX, cartesian=False)
            self.t0 = self.t

        elif self.mode == Mode.JOINT and self.t - self.t0 > self.tmove:
            self.mode = Mode.TASK
            self.t0 = self.t
            self.TS['p0'] = RobotPose(*(self.chain.fkin(self.JS['q_last'])[:2]))
            self.tmove = splinetime(self.TS['p0'].x, self.TS['goal'].x, v0=0, vf=0)
            # ros_print(self, f"\n\n\n\n\n\n\nTMOVE: {self.tmove}\n\n\n\n\n\n\n")
            # setup splinetime

        elif self.mode == Mode.TASK and self.t - self.t0 > self.tmove:
            self.mode = Mode.STILL

        # ros_print(self, self.mode)
        
        
        self.cmdmsg.position = list(q) + [nan]
        self.cmdmsg.velocity = list(qdot) + [nan]
        self.cmdmsg.effort = list(qeff) + [float(self.grip)]
        self.cmdpub.publish(self.cmdmsg)
        boolmsg = Bool()
        boolmsg.data = ready
        self.ready.publish(boolmsg)


# mais je suis francais oh honhonhonhonhon
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('low_level')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
