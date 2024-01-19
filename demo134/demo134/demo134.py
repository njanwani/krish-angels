#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import rclpy

from rclpy.node         import Node
from sensor_msgs.msg    import JointState


#
#   Definitions
#
RATE = 100.0            # Hertz


#
#   DEMO Node Class
#
class DemoNode(Node):
    # Initialization.
    
    AMPS = np.array([0.4, 0.3, 0.25])
    PERIODS = np.array([1.0, 4.0, 2.0])
    WAVE_T = 6
    VMAX = 2
    AMAX = VMAX / 3
    
    def __init__(self, name):
        
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.t = 0
        self.t0 = self.t
        self.mode = 0
        self.position0 = self.grabfbk()
        self.Tmove = DemoNode.splinetime(self.position0, DemoNode.wave(0)[0], np.zeros(3), DemoNode.wave(0)[1])
        # Create a temporary subscriber to grab the initial position.
        self.get_logger().info("Initial positions: %r" % self.position0)

        # Create a message and publisher to send the joint commands.
        self.cmdmsg = JointState()
        self.cmdpub = self.create_publisher(JointState, '/joint_commands', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

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
        pass


    # Send a command - called repeatedly by the timer.
    def sendcmd(self):
        # Build up the message and publish.
        pos = np.zeros(3)
        vel = np.zeros(3)
        tau = self.t - self.t0
        if self.mode == 0:
            init_pos = self.position0
            pos, vel = DemoNode.spline(tau, self.Tmove, init_pos, DemoNode.wave(0)[0], np.zeros(3), DemoNode.wave(0)[1])
        elif self.mode == 1:
            pos, vel = DemoNode.wave(tau)
        elif self.mode == 2:
            pos, vel = DemoNode.spline(tau, self.Tmove, DemoNode.wave(0)[0], self.position0, DemoNode.wave(0)[1], np.zeros(3))
        elif self.mode == 3:
            pos, vel = self.position0, np.zeros(3)
        else:
            raise Exception('Unknown mode encountered')
        
        if self.t - self.t0 > self.Tmove:
            self.mode += 1
            self.mode %= 4 
            self.t0 = self.t
            if self.mode == 0:
                self.Tmove = DemoNode.splinetime(self.position0, DemoNode.wave(0)[0], np.zeros(3), DemoNode.wave(0)[1])
            elif self.mode == 1:
                self.Tmove = 6
            elif self.mode == 2:
                # self.position0 = np.zeros(3)
                # self.position0[1] = np.pi
                self.Tmove = DemoNode.splinetime(DemoNode.wave(0)[0], self.position0, DemoNode.wave(0)[1], np.zeros(3))
            elif self.mode == 3:
                self.Tmove = 1
            else:
                raise Exception('Unknown mode encountered')
        
        self.get_logger().info(str(vel))
        
        self.cmdmsg.header.stamp = self.get_clock().now().to_msg()
        self.cmdmsg.name         = ['one', 'two', 'three']
        self.cmdmsg.position     = list(pos)
        self.cmdmsg.velocity     = list(vel)
        self.cmdmsg.effort       = list(np.zeros(3))
        self.cmdpub.publish(self.cmdmsg)
        self.t += 0.01

    def wave(tau):
        pos = DemoNode.AMPS * np.sin(tau * np.pi / 6 * 2 * DemoNode.PERIODS) + np.array([0, 0, np.pi/2])
        vel = np.multiply(DemoNode.AMPS, 2 * np.pi / 6 * DemoNode.PERIODS) * np.cos(tau * np.pi / 6 * 2 * DemoNode.PERIODS)
        return pos, vel

    def spline(t, T, p0, pf, v0, vf):
        # Compute the parameters.
        a = p0
        b = v0
        c =   3*(pf-p0)/T**2 - vf/T    - 2*v0/T
        d = - 2*(pf-p0)/T**3 + vf/T**2 +   v0/T**2
        # Compute the current (p,v).
        p = a + b * t +   c * t**2 +   d * t**3
        v =     b     + 2*c * t    + 3*d * t**2
        return (p,v)
    
    def splinetime(p0, pf, v0, vf):
        m = max(1.5 * (np.linalg.norm(pf - p0) / DemoNode.VMAX + np.abs(v0) / DemoNode.AMAX + np.abs(vf) / DemoNode.AMAX))
        return max(m, 0.5)
    


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
