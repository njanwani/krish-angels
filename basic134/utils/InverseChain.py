'''hw6p4sol.py

   This is the solution code for HW6 Problem 4.

   This combines position and orientation movements.  It moves (a)
   from the initial position to the starting point, then (b) up/down
   while rotating the tip (cube).

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from utils.TransformHelpers   import *
from utils.TrajectoryUtils    import *
from utils.KinematicChain  import KinematicChain


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        # Define the various points.
        self.q0 = np.radians(np.array([0, 90, -90, 0, 0, 0]).reshape((-1,1)))
        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((-1,1))
        self.R0 = Reye()

        self.plow  = np.array([0.0, 0.5, 0.3]).reshape((-1,1))
        self.phigh = np.array([0.0, 0.5, 0.9]).reshape((-1,1))

        # Initialize the current/starting joint position and set the
        # desired tip position/orientation to match.
        self.q  = self.q0
        self.pd = self.p0
        self.Rd = self.R0
        # (self.pd, self.Rd, _, _) = self.chain.fkin(self.q)

        # Pick the convergence bandwidth.
        self.lam = 20

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # End after 11s.
        if t>11:
            return None

        # Decide which phase we are in:
        if t < 3.0:
            # Approach movement:
            (s0, s0dot) = goto(t, 3.0, 0.0, 1.0)

            pd = self.p0 + (self.plow - self.p0) * s0
            vd =           (self.plow - self.p0) * s0dot

            Rd = Rotz(-pi/2 * s0)
            wd = ez() * (-pi/2 * s0dot)

        else:
            # Pre-compute the path variables.  To show different
            # options, we compute the position path variable using
            # sinusoids and the orientation variable via splines.
            sp    =      - cos(pi/2 * (t-3.0))
            spdot = pi/2 * sin(pi/2 * (t-3.0))

            t1 = (t-3) % 8.0
            if t1 < 4.0:
                (sR, sRdot) = goto(t1,     4.0, -1.0,  1.0)
            else:
                (sR, sRdot) = goto(t1-4.0, 4.0,  1.0, -1.0)

            # Use the path variables to compute the trajectory.
            pd = 0.5*(self.phigh+self.plow) + 0.5*(self.phigh-self.plow) * sp
            vd =                            + 0.5*(self.phigh-self.plow) * spdot

            Rd = Rotz(pi/2 * sR)
            wd = ez() * (pi/2 * sRdot)

        # Grab the last joint value and desired orientation.
        qlast  = self.q
        pdlast = self.pd
        Rdlast = self.Rd

        # Compute the old forward kinematics.
        (p, R, Jv, Jw) = self.chain.fkin(qlast)

        # Compute the inverse kinematics
        vr    = vd + self.lam * ep(pdlast, p)
        wr    = wd + self.lam * eR(Rdlast, R)
        J     = np.vstack((Jv, Jw))
        xrdot = np.vstack((vr, wr))
        qdot  = np.linalg.pinv(J) @ xrdot

        # Integrate the joint position.
        q = qlast + dt * qdot

        # Save the joint value and desired values for next cycle.
        self.q  = q
        self.pd = pd
        self.Rd = Rd

        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())

