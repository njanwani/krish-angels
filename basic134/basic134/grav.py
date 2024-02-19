#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import rclpy
import time
# from KinematicChain import KinematicChain as KC
from utils.KinematicChain import KinematicChain as KC
from utils.TrajectoryUtils import *
from utils.TransformHelpers import *
from enum import Enum
from utils.pyutils import *

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float32

#
#   Definitions
#
RATE = 100.0            # Hertz


GRAV_SPLINE = 0
TASK_SPLINE = 1
CHILL = 2

#
#   DEMO Node Class
#
class DemoNode(Node):

    VMAX    = 0.8
    AMAX    = VMAX / 10
    THOLD   = 1

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
        # Create a temporary subscriber to grab the initial position.
        self.position0      = self.grabfbk()
        # self.position0 = np.array([0, -np.pi / 2, np.pi / 2, 0, 0])
        self.p0, self.R0, _, _ = self.chain.fkin(self.position0)
        self.get_logger().info("Initial positions: %r" % self.position0)
        self.t = 0
        self.t0 = self.t
        self.tmove = 1
        self.mode = 'GRAV'

        # current robot position IN JOINT SPACE
        self.curr_pos = self.position0
        self.goal_position = np.array([0.3, 0.3, 0.05]).reshape((3,1))
        self.R_goal = Roty(0)
        self.rate = 0.01

        self.T_filter = 0.5
        self.t0_filter = time.time()
        self.filter_effort = 0
        # Subscribe to the actual joint states, waiting for the first message.
        self.actpos = None
        self.acteffort = None
        self.statessub = self.create_subscription(JointState, '/joint_states', self.cb_states, 1)
        while self.actpos is None:
            rclpy.spin_once(self)
        self.get_logger().info("Initial positions: %r" % self.actpos)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)
        # self.tip_marker = self.create_publisher(Marker, '/tip_marker', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a subscriber to receive point messages.
        self.fbksub = self.create_subscription(Float32, '/B', self.recvB, 10)

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))
    
    # Save the actual position.
    def cb_states(self, msg):
        dt = time.time() - self.t0_filter
        self.t0_filter = time.time()
        self.actpos = msg.position
        self.acteffort = msg.effort
        self.filter_effort += dt / self.T_filter * (np.array(msg.effort) - self.filter_effort)
    
    def gravity(self, pos):
        tau_shoulder = float(self.B * np.cos(self.actpos[1]) + max(0, self.C * np.cos(self.actpos[1]) * np.cos(self.actpos[2])))
        return [0.0, tau_shoulder, 0.0]
    
    def contact(self):
        if np.any(self.filter_effort > 1.5):
            return True
        
        return False

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
    
    # Receive feedback - called repeatedly by incoming messages.
    def recvfbk(self, fbkmsg):
        # Just print the position (for now).
        # print(list(fbkmsg.position))
        self.curr_pos = fbkmsg.position
        
    # Receive a float message - called by incoming messages.
    def recvB(self, floatmsg: Float32):
        # Extract the data.
        self.B = floatmsg.data

    def compute_ts_spline(self, s, sdot):
        # p_last, _ = spline5(self.t - self.t0,
        #                     self.tmove,
        #                     self.chain.fkin(self.q0)[0].flatten(),
        #                     self.queue[0],
        #                     0, 0, 0, 0)
        # _, v = spline5(self.t + 1.0 / RATE - self.t0,
        #                self.tmove,
        #                self.chain.fkin(self.q0)[0].flatten(),
        #                self.queue[0],
        #                0, 0, 0, 0)



        p = pinter(self.p0, self.goal_position, s)
        v = vinter(self.p0, self.goal_position, sdot)

        R = Rinter(self.R0, self.R_goal, s)
        w = winter(self.R0, self.R_goal, sdot)
        
        q, qdot = self.chain.ikin(abs(self.rate), self.actpos, p, v, w, R)
        return q, qdot

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):

        self.t += self.rate
        ros_print(self,f'{self.t - self.t0} {self.mode}\n\n\n')
        # if not (self.rate <= 0):
        #     self.h, self.hdot = spline5(self.t, self.tmove, 0, 1, 0, 0, 0, 0)
        # else:
        #     self.h, self.hdot = spline5(self.tmove - self.t, self.tmove, 1, 0, 0, 0, 0, 0)

        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow', 'wrist', 'end']
        T = 2 # seconds
        nan = float("nan")
        
        if self.mode == 'GRAV':
            t_elbow = 6 * np.cos(self.actpos[1] - self.actpos[2])
            t_shoulder = -t_elbow - 5.2 * np.cos(self.actpos[1]) #5.3
            q = [nan] * 5
            qdot = [nan] * 5
            effort = spline5(self.t - self.t0, self.tmove, 0, 1, 0, 0, 0, 0)[0] * np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])
        elif self.mode == 'TASK':
            s, sdot = spline5(self.t - self.t0, self.tmove, 0, 1, 0, 0, 0, 0)
            q, qdot = self.compute_ts_spline(s, sdot)
            t_elbow = 6 * np.cos(q[1] - q[2])
            t_shoulder = -t_elbow - 5.2 * np.cos(q[1]) #5.3
            effort = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])
        elif self.mode == 'CHILL':
            q = [nan] * 5
            qdot = [nan] * 5
            t_elbow = 6 * np.cos(self.actpos[1] - self.actpos[2])
            t_shoulder = -t_elbow - 5.2 * np.cos(self.actpos[1]) #5.3
            effort = np.array([0.0, t_shoulder, t_elbow, 0.0, 0.0])
        else:
            raise Exception('OOPS')
        
        self.qlast = q

        if self.mode == 'GRAV' and self.t - self.t0 > self.tmove:
            ros_print(self, 'SWITCHSWITCH\n\n\n\n\n\n\n')
            self.mode = 'CHILL'
            self.tmove = 6
            self.t0 = self.t
        elif self.mode == 'TASK' and self.t - self.t0 > self.tmove:
            self.mode = 'CHILL'


        
        self.cmdmsg.position = list(q)
        self.cmdmsg.velocity = list(qdot)
        self.cmdmsg.effort = list(effort)
        self.cmdpub.publish(self.cmdmsg)


# mais je suis francais oh honhonhonhonhon
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
