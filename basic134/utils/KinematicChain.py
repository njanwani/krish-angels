import enum
import rclpy
import numpy as np

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from std_msgs.msg               import String
from urdf_parser_py.urdf        import Robot

# Grab the utilities
from utils.TransformHelpers   import *

class Joint(enum.Enum):
    FIXED    = 0
    REVOLUTE = 1
    LINEAR   = 2

class KinematicStep():
    def __init__(self, Tshift, elocal, type, name):
        # Store the permanent/fixed/URDF data.
        self.Tshift = Tshift    # Transform w.r.t. previous frame
        self.elocal = elocal    # Joint axis in the local frame
        self.type   = type      # Joint type
        self.name   = name      # Joint name
        self.dof    = None      # Joint DOF number (or None if FIXED)

        # Clear the information to be updated every walk up the chain.
        self.clear()

    def clear(self):
        self.T = None           # Transform of frame w.r.t. world
        self.p = None           # Position  of frame w.r.t. world
        self.R = None           # Rotation  of frame w.r.t. world
        self.e = None           # Axis vector        w.r.t. world

    @classmethod
    def FromRevoluteJoint(cls, joint):
        return KinematicStep(T_from_URDF_origin(joint.origin),
                             e_from_URDF_axis(joint.axis),
                             Joint.REVOLUTE, joint.name)
    @classmethod
    def FromLinearJoint(cls, joint):
        return KinematicStep(T_from_URDF_origin(joint.origin),
                             e_from_URDF_axis(joint.axis),
                             Joint.LINEAR, joint.name)
    @classmethod
    def FromFixedJoint(cls, joint):
        return KinematicStep(T_from_URDF_origin(joint.origin),
                             np.zeros((3,1)),
                             Joint.FIXED, joint.name)

# Define the full kinematic chain
class KinematicChain():
    # Helper functions for printing info and errors.
    def info(self, string):
        self.node.get_logger().info("KinematicChain: " + string)
    def error(self, string):
        self.node.get_logger().error("KinematicChain: " + string)
        raise Exception(string)

    # Initialization.
    def __init__(self, node, baseframe, tipframe, expectedjointnames, lam=20):
        # Store the node (for the helper functions).
        self.node = node
        self.lam = lam
        # Prepare the information.
        self.steps = []
        self.dofs  = 0

        # Grab the info from the URDF!
        self.load(baseframe, tipframe, expectedjointnames)

    # Load the info from the URDF.
    def load(self, baseframe, tipframe, expectedjointnames):
        # Create a temporary subscriber to receive the URDF.  We use
        # the TRANSIENT_LOCAL durability, so that we see the last
        # message already published (if any).
        self.info("Waiting for the URDF to be published...")
        self.urdf = None
        def cb(msg):
            self.urdf = msg.data
        topic   = '/robot_description'
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        sub = self.node.create_subscription(String, topic, cb, quality)
        while self.urdf is None:
            rclpy.spin_once(self.node)
        self.node.destroy_subscription(sub)

        # Convert the URDF string into a Robot object and report.
        robot = Robot.from_xml_string(self.urdf)
        self.info("Proccessing URDF for robot '%s'" % robot.name)

        # Parse the Robot object into a list of kinematic steps from
        # the base frame to the tip frame.  Search backwards, as the
        # robot could be a tree structure: while a parent may have
        # multiple children, every child has only one parent.  The
        # resulting chain of steps is unique.
        frame = tipframe
        while (frame != baseframe):
            # Look for the URDF joint to the parent frame.
            joint = next((j for j in robot.joints if j.child == frame), None)
            if (joint is None):
                self.error("Unable find joint connecting to '%s'" % frame)
            if (joint.parent == frame):
                self.error("Joint '%s' connects '%s' to itself" %
                           (joint.name, frame))
            frame = joint.parent

            # Convert the URDF joint into a simple step.
            if joint.type == 'revolute' or joint.type == 'continuous':
                self.steps.insert(0, KinematicStep.FromRevoluteJoint(joint))
            elif joint.type == 'prismatic':
                self.steps.insert(0, KinematicStep.FromLinearJoint(joint))
            elif joint.type == 'fixed':
                self.steps.insert(0, KinematicStep.FromFixedJoint(joint))
            else:
                self.error("Joint '%s' has unknown type '%s'" %
                           (joint.name, joint.type))

        # Set the active DOF numbers walking up the steps.
        dof = 0
        for s in self.steps:
            if s.type is not Joint.FIXED:
                s.dof = dof
                dof += 1
        self.dofs = dof
        self.info("URDF has %d steps, %d active DOFs:" %
                  (len(self.steps), self.dofs))

        # Report we found.
        for (step, s) in enumerate(self.steps):
            string = "Step #%d %-8s " % (step, s.type.name)
            string += "      " if s.dof is None else "DOF #%d" % s.dof
            string += " '%s'" % s.name
            self.info(string)

        # Confirm the active joint names matches the expectation
        jointnames = [s.name for s in self.steps if s.dof is not None]
        if jointnames != list(expectedjointnames):
            self.error("Chain does not match the expected names: " +
                  str(expectedjointnames))


    # Compute the forward kinematics!
    def fkin(self, q):
        # Check the number of joints
        if (len(q) != self.dofs):
            self.error(f"Number of joint angles {len(q)} does not chain {self.dofs}")

        # Clear any data from past invocations (just to be safe).
        for s in self.steps:
            s.clear()

        # Initialize the T matrix to walk up the chain, w.r.t. world frame!
        T = np.eye(4)

        # Walk the chain, one step at a time.  Record the T transform
        # w.r.t. world for each step.
        for s in self.steps:
            # Always apply the shift.
            T = T @ s.Tshift

            # For active joints, also apply the joint movement.
            if s.type is Joint.REVOLUTE:
                # Revolute is a rotation:
                T = T @ T_from_Rp(Rote(s.elocal, q[s.dof]), pzero())
            elif s.type is Joint.LINEAR:
                # Linear is a translation:
                T = T @ T_from_Rp(Reye(), s.elocal * q[s.dof])

            # Store the info (w.r.t. world frame) into the step.
            s.T = T
            s.p = p_from_T(T)
            s.R = R_from_T(T)
            s.e = R_from_T(T) @ s.elocal

        # Collect the tip information.
        ptip = p_from_T(T)
        Rtip = R_from_T(T)

        # Re-walk up the chain to fill in the Jacobians.
        Jv = np.zeros((3,self.dofs))
        Jw = np.zeros((3,self.dofs))
        for s in self.steps:
            if s.type is Joint.REVOLUTE:
                # Revolute is a rotation:
                Jv[:,s.dof:s.dof+1] = cross(s.e, ptip - s.p)
                Jw[:,s.dof:s.dof+1] = s.e
            elif s.type is Joint.LINEAR:
                # Linear is a translation:
                Jv[:,s.dof:s.dof+1] = s.e
                Jw[:,s.dof:s.dof+1] = np.zeros((3,1))

        # Return the info
        return (ptip, Rtip, Jv, Jw)
    
    def weighted_inv(J, gamma):
        return J.T @ np.linalg.pinv(J @ J.T + gamma**2 * np.identity(J.shape[0]))
    
    def ikin(self, dt, qlast, pdlast, vd, q_secondary=np.zeros(3), lam_secondary=1):
        # Compute the old forward kinematics.
        (p, R, Jv, Jw) = self.fkin(qlast)
        q_secondary = lam_secondary * (np.array([np.pi / 2, -np.pi / 2, np.pi / 2]).reshape((3,1)) - qlast)
        # Compute the inverse kinematics
        vr    = vd + self.lam * ep(pdlast, p)
        # wr    = wd + self.lam * eR(Rdlast, R)
        J     = np.vstack((Jv))
        xrdot = np.vstack((vr))
        qdot  = np.linalg.pinv(J) @ xrdot #+ (1 - np.linalg.pinv(J) @ J) @ q_secondary.reshape((3,1)) #KinematicChain.weighted_inv(J, 0.5) @ xrdot
        # raise Exception(f'{pdlast}\n\n\n{p}\n\n\n{vd}\n\n\n{J}\n\n\n{xrdot}\n\n\n{qdot}\n\n\nZamnnnn')
        # Integrate the joint position.
        q = qlast + dt * qdot
        
        return (q.flatten(), qdot.flatten())


# point 1: x: 0.0095 y: -0.147  z: 0.0
# point 2: x: 0.385  y: -0.197  z: 0.0
# point 3: x: 0.147  y: 0.276   z: 0.0