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
from enum import Enum

from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point


#
#   Definitions
#
RATE = 100.0            # Hertz

class Mode(Enum):
    JOINT_SPLINE      = 0
    TASK_SPLINE       = 1
    HOLD              = 2

#
#   DEMO Node Class
#
class DemoNode(Node):

    VMAX    = 0.8
    AMAX    = VMAX / 10
    THOLD   = 1

    POINT_LIB = (np.array([0.51, .1725, 0.01]),         # side x
                 np.array([0.30, -0.00895, 0.007644]),  # middle x
                 np.array([0, 0, 50.0]),                # out of task space + singularity
                 np.array([1.5, .1725, 0.01]),          # out of task space
    )

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        self.chain = KC(self, 'world', 'tip', ['base', 'shoulder', 'elbow'])
        # Create a temporary subscriber to grab the initial position.
        self.position0      = self.grabfbk()
        self.get_logger().info("Initial positions: %r" % self.position0)

        # current robot position IN JOINT SPACE
        self.curr_pos = self.position0

        # spline coordinates IN CARTESIAN SPACE
        self.joint_wait = np.array([0, np.pi/2, -np.pi/2])
        self.position_init = self.chain.fkin(self.position0)[0].flatten()
        self.mag_zeropos = np.linalg.norm(self.chain.fkin(np.zeros(3))[0].flatten())
        self.position_wait = self.chain.fkin(self.joint_wait)[0].flatten() #np.array([0.29972, -0.0508, 0.50607])
        self.get_logger().info("Wait positions: %r" % self.position_wait)
        # self.positions = [self.position_wait] + [bound_taskspace(DemoNode.POINT_LIB[i], mag_zeropos) for i in range(len(DemoNode.POINT_LIB))]
        self.queue = [self.position_wait]
        self.qlast = np.array(self.curr_pos).reshape((3,1))

        self.t = time.time()
        self.t0 = self.t
        self.tmove = splinetime(self.position0, self.joint_wait, np.zeros(3), np.zeros(3), DemoNode.VMAX, DemoNode.AMAX, cartesian=False)
        self.spline_index = -1
        self.spline_segs = lambda idx: (self.queue[idx], self.queue[(idx + 1) % len(self.positions)])
        self.mode = Mode.JOINT_SPLINE

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
        self.fbksub = self.create_subscription(Point, '/point', self.recvpoint, 10)

        # Create a subscriber to continually receive joint state messages.
        self.fbksub = self.create_subscription(
            JointState, '/joint_states', self.recvfbk, 10)

        # Create a timer to keep calculating/sending commands.
        rate       = RATE
        self.timer = self.create_timer(1/rate, self.sendcmd)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))


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
        
    # Receive a point message - called by incoming messages.
    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        self.queue.append(bound_taskspace(np.array([x,y,z]), self.mag_zeropos))
        self.queue.append(self.position_wait)
        # self.spline_index += 1
        # self.spline_index %= len(self.positions)
        if self.mode == Mode.JOINT_SPLINE:
            return
        self.mode = Mode.TASK_SPLINE
        self.t0 = self.t
        self.tmove = splinetime(*self.queue[:2],
                                np.zeros(3),
                                np.zeros(3),
                                DemoNode.VMAX,
                                DemoNode.AMAX)
        # Report.
        self.get_logger().info("Running point %r, %r, %r" % (x,y,z))

    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        self.t = time.time()
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['base', 'shoulder', 'elbow']
        if self.mode == Mode.TASK_SPLINE:
            p_last, _ = spline5(self.t - self.t0, self.tmove, 
                        *self.queue[:2], 0, 0, 0, 0)
            _, v = spline5(self.t + 1.0 / RATE - self.t0, self.tmove, 
                        *self.queue[:2], 0, 0, 0, 0)
            
            # self.get_logger().info(str(p_last) + '\n\n\n' + str(v) + '\n\n\n' + str(self.qlast))

            q, qdot = self.chain.ikin(1.0 / RATE, np.array(self.qlast).reshape((3,1)), p_last.reshape((3,1)), v.reshape((3,1)))
        elif self.mode == Mode.HOLD:
            q, qdot = self.qlast, np.zeros(3)
        elif self.mode == Mode.JOINT_SPLINE:
            q, qdot = spline5(self.t - self.t0, self.tmove, self.position0, self.joint_wait, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        else:
            self.get_logger().info("I fucked up bad...")
            raise Exception("Zamn")

        if self.mode == Mode.TASK_SPLINE and self.t - self.t0 > self.tmove:
            self.queue.pop(0)
            self.mode = Mode.HOLD
            self.t0 = self.t
            self.tmove = DemoNode.THOLD
        elif self.mode == Mode.HOLD and self.t - self.t0 > self.tmove and self.t - self.t0 > self.tmove and len(self.queue) > 1:
            self.mode = Mode.TASK_SPLINE
            self.t0 = self.t
            self.tmove = splinetime(*self.queue[:2],
                                    np.zeros(3),
                                    np.zeros(3),
                                    DemoNode.VMAX,
                                    DemoNode.AMAX)
        elif self.mode == Mode.JOINT_SPLINE and self.t - self.t0 > self.tmove:
            self.mode = Mode.HOLD
            self.t0 = self.t
            self.tmove = DemoNode.THOLD

        self.qlast = q
        self.cmdmsg.position = q.flatten().tolist()
        self.cmdmsg.velocity = qdot.flatten().tolist()
        self.cmdmsg.effort = [0.0, 0.0, 0.0]
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
