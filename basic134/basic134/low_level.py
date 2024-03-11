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
    def __init__(self, x, theta, pitch=0, R=None):
        self.x = x
        self.theta = theta
        self.pitch = pitch
        if R is None:
            self.R = Reye() @ Roty(self.pitch) @ Rotz(self.theta)

    def recalcR(self):
        self.R = Reye() @ Roty(self.pitch) @ Rotz(self.theta)

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

    GRIP_MIN = -3
    GRIP_MAX = 0

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
        self.grip_cmd = 0
        self.grip_act = 0
        self.grip_0 = 0
        self.position0 = self.grabfbk()
        self.grip_0 = self.position0[-1]
        self.position0 = self.position0[:-1]
        self.queue = []

        self.t = time.time()
        self.tlast = None
        self.t0 = self.t
        self.tmove = 2
        self.mode = Mode.START
        self.lastmode = None

        self.JS = {}
        self.JS['q0'] = np.array(self.position0)
        self.JS['q_act'] = None
        self.JS['q_last'] = self.JS['q0']
        self.JS['q_start'] = np.array([np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0, 0.0])

        self.TS = {}
        self.TS['p0'] = RobotPose(self.chain.fkin(self.position0)[0], self.position0[0] - self.position0[-1])
        self.TS['goal'] = RobotPose(self.chain.fkin(self.JS['q_start'])[0], self.JS['q_start'][0] - self.JS['q_start'][-1])
        self.TS['v0'] = 0
        self.TS['a0'] = 0

        self.s_last = 0
        self.sdot_last = 0
        self.sdotdot_last = 0

        self.effort = Filter(1, 0)
        

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(JointState, '/joint_states', self.cb_states, 10)
        self.grip_sub = self.create_subscription(Bool, '/low_level/grip', self.cb_grip, 10)
        
        ros_print(self, 'waiting for feedback')
        while self.JS['q_act'] is None:
            rclpy.spin_once(self)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        self.ready = self.create_publisher(Bool, '/' + name + '/ready', 10)
        self.armed_pub = self.create_publisher(Bool, '/' + name + '/armed', 10)
        self.armed = False

        while not self.count_subscribers('/joint_commands'):
            pass

        # Create a subscriber to receive point messages.
        self.recvpt_sub = self.create_subscription(Pose, f'/{name}/goal_2', self.recvpt, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1 / rate, self.sendcmd)
        ros_print(self, 'Node started')
    
    def cb_grip(self, msg: Bool):
        changed = False
        offset_z = 0.03
        if msg.data and self.grip_cmd != self.GRIP_MIN:
            self.grip_cmd = self.GRIP_MIN
            changed = True
            
        elif not msg.data and self.grip_cmd != self.GRIP_MAX:
            self.grip_cmd = self.GRIP_MAX
            changed = True
            offset_z *= -1
        
        if changed:
            pose = Pose()
            pose.position.x = float(self.TS['goal'].x[0])
            pose.position.y = float(self.TS['goal'].x[1])
            pose.position.z = float(self.TS['goal'].x[2] + offset_z)
            pose.orientation.z = float(self.TS['goal'].theta)

            self.recvpt(pose, gripping=True)



    def state_space(self):
        if self.mode == Mode.STILL:
            return 0,0,0
        
        return spline5(self.t - self.t0, self.tmove, p0=self.s_last,
                                                     pf=1,
                                                     v0=self.sdot_last,
                                                     vf=0,
                                                     a0=self.sdotdot_last,
                                                     af=0)

    def recvpt(self, msg: Pose, gripping = False):
        msg.position.y = msg.position.y #+ 0.026 - 0.005 * msg.position.y
        msg.position.x = msg.position.x #- 0.002 - 0.03 * msg.position.y
        msg.position.z = msg.position.z #+ 0.02 * msg.position.x
        # ros_print(self, msg.position.y)
        if np.all(self.TS['goal'].x == self.TS['p0'].x) and np.all(self.TS['goal'].R == self.TS['p0'].R):
            return

        reach = (msg.position.x**2 + msg.position.y**2)**0.5
        if reach < 0.1 or reach > 0.7 or msg.position.z < 0.0 or msg.position.z > 0.51:
            # ros_print(self, f'{msg.position.x, msg.position.y}')
            return
        
        R = Reye() @ Roty(msg.orientation.y) @ Rotz(msg.orientation.z)
        
        if np.all(np.isclose(np.array([msg.position.x, msg.position.y, msg.position.z]), self.TS['goal'].x, atol=0.01)) \
           and np.all(np.isclose(R, self.TS['goal'].R, atol=0.01)) \
           and not gripping:
            return
        self.armed = False
        ros_print(self, 'RE_SPLINE')
        v_last, a_last = 0,0
        if self.mode == Mode.TASK:
            _, v_last, a_last = spline5(self.t - self.t0, self.tmove, self.TS['p0'].x.flatten(), self.TS['goal'].x.flatten(), self.TS['v0'],0, self.TS['a0'], 0)
            v_last, a_last = v_last.flatten(), a_last.flatten()
            
        self.TS['goal'].x = np.array([msg.position.x,
                                      msg.position.y,
                                      msg.position.z])
        self.TS['goal'].theta = msg.orientation.z
        self.TS['goal'].pitch = msg.orientation.y
        self.TS['goal'].recalcR()
        self.TS['p0'] = RobotPose(self.chain.fkin(self.JS['q_last'])[0], self.JS['q_last'][0] - self.JS['q_last'][-1])
        self.TS['p0'].R = self.chain.fkin(self.JS['q_last'])[1]
        # -self.JS['q_last'][1] + self.JS['q_last'][1] - self.JS['q_last'][1]

        if self.mode in [Mode.STILL, Mode.TASK]:
            self.mode = Mode.TASK
            self.t0 = self.t
            self.grip_0 = self.grip_act
            self.tmove = max(splinetime(self.TS['p0'].x, self.TS['goal'].x, v0=v_last, vf=0), int(gripping))
            self.s_last = 0
            self.TS['v0'] = v_last
            self.TS['a0'] = a_last
    
    # Save the actual position.
    def cb_states(self, msg: JointState):
        self.grip_act = msg.position[-1]
        self.JS['q_act'] = np.array(msg.position)[:-1]
        if self.mode == Mode.START:
            self.JS['q0'] = self.JS['q_act']
        self.effort.update(msg.effort[:-1])
    
    def gravity(self, q):
        # ros_print(self, q[3] - q[2] + q[1])
        t_wrist = 0.9 * np.sin(q[3] - q[2] + q[1]) + 0.1 * np.cos(q[-1])
        t_elbow = 7.5 * np.cos(q[1] - q[2]) - t_wrist #5.65 
        t_shoulder = -t_elbow - 9.0 * np.cos(q[1])
        return t_shoulder, t_elbow, t_wrist

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
        theta, thetadot, _ = spline5(self.t - self.t0, self.tmove, self.TS['p0'].theta, self.TS['goal'].theta, 0, 0, 0, 0)
        R = Rinter(self.TS['p0'].R, self.TS['goal'].R, s)
        w = winter(self.TS['p0'].R, self.TS['goal'].R, sdot)
        gq, gqdot, _ = spline5(self.t - self.t0, self.tmove, self.grip_0, self.grip_cmd, 0, 0, 0, 0)

        # ros_print(self, f'p {p}')
        
        q, qdot, sing = self.chain.ikin(1 / RATE, self.JS['q_last'], p.reshape((3,1)), v.reshape((3,1)), w, R, tip_angle=theta, tip_dot=thetadot)
        q, qdot = np.array(list(q) + [gq]), np.array(list(qdot) + [gqdot])
        if sing:
            mode = Mode.JOINT
            self.t0 = self.t
            self.JS['q0'] = self.JS['q_act']
            self.tmove = splinetime(self.JS['q0'], self.JS['q_start'], 0, 0, DemoNode.VMAX, DemoNode.AMAX, cartesian=False)
        return q, qdot

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # ros_print(self, self.mode)
        if self.tlast is None:
            self.tlast = time.time() - 1 / RATE
        dt = time.time() - self.tlast
        self.t += dt
        self.tlast = time.time()
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'end', 'grip']
        nan = float("nan")
        ready = False
        
        q, qdot, qeff = None, None, None
        if self.mode == Mode.START:
            q = self.JS['q0']
            q = np.append(q, [0])
            qdot = np.zeros(6)
            t_shoulder, t_elbow, t_wrist = self.gravity(self.JS['q_act'])
            qeff = spline5(self.t - self.t0, self.tmove, 0.0, 1, 0, 0, 0, 0)[0] * np.array([0.0, t_shoulder, t_elbow, t_wrist, 0.0, nan])

        elif self.mode == Mode.GRAV:
            q = [nan] * 6
            qdot = [nan] * 6
            t_shoulder, t_elbow, t_wrist = self.gravity(self.JS['q_act'])
            qeff = np.array([0.0, t_shoulder, t_elbow, t_wrist, 0.0, 0.0])

        elif self.mode == Mode.TASK:
            s, sdot, _ = self.state_space()
            q, qdot = self.compute_ts_spline(s, sdot)
            t_shoulder, t_elbow, t_wrist = self.gravity(q)
            qeff = np.array([0.0, t_shoulder, t_elbow, t_wrist, 0.0, 0.0])
            ready = True
            self.armed = False

        elif self.mode == Mode.JOINT:
            q, qdot, _ = spline5(self.t - self.t0, self.tmove, self.JS['q0'], self.JS['q_start'], 0, 0, 0, 0)
            q, qdot = np.array(list(q) + [self.grip_cmd]), np.array(list(qdot) + [self.grip_cmd])
            t_shoulder, t_elbow, t_wrist = self.gravity(q)
            qeff = 1 * np.array([0.0, t_shoulder, t_elbow, t_wrist, 0.0, 0.0])

        elif self.mode == Mode.STILL:
            q, qdot = self.JS['q_last'], np.zeros(5)
            q, qdot = np.array(list(q) + [self.grip_cmd]), np.array(list(qdot) + [0])
            t_shoulder, t_elbow, t_wrist = self.gravity(q)
            qeff = np.array([0.0, t_shoulder, t_elbow, t_wrist, 0.0, 0.0])
            ready = True
        
        else:
            raise Exception('OOPS')
        
        self.JS['q_last'] = q[:-1]
        self.JS['qd_last'] = qdot[:-1]

        if self.mode == Mode.START and self.t - self.t0 > self.tmove:
            self.lastmode = self.mode
            self.mode = Mode.JOINT
            self.tmove = splinetime(self.JS['q0'], self.JS['q_start'], 0, 0, DemoNode.VMAX, DemoNode.AMAX, cartesian=False)
            self.t0 = self.t

        elif self.mode == Mode.JOINT and self.t - self.t0 > self.tmove:
            self.lastmode = self.mode
            self.mode = Mode.STILL
            self.t0 = self.t
            self.TS['p0'] = RobotPose(self.chain.fkin(self.JS['q_last'])[0], self.JS['q_last'][0] - self.JS['q_last'][-1])
            self.tmove = splinetime(self.TS['p0'].x, self.TS['goal'].x, v0=0, vf=0)
            # ros_print(self, f"\n\n\n\n\n\n\nTMOVE: {self.tmove}\n\n\n\n\n\n\n")
            # setup splinetime

        elif self.mode == Mode.TASK and self.t - self.t0 > self.tmove:
            self.lastmode = self.mode
            self.armed = True
            self.mode = Mode.STILL
        
        # q = q + np.array([0, 0, 0, 0, 0])
        
        self.cmdmsg.position = list(q)
        self.cmdmsg.velocity = list(qdot)
        self.cmdmsg.effort = list(qeff)
        self.cmdpub.publish(self.cmdmsg)
        boolmsg = Bool()
        boolmsg.data = ready
        self.ready.publish(boolmsg)
        
        boolmsg = Bool()
        boolmsg.data = self.armed
        self.armed_pub.publish(boolmsg)



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
