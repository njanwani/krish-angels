#!/usr/bin/env python3
#
#   demo134.py
#
#   Demonstration node to interact with the HEBIs.
#
import numpy as np
import serial
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

'''
STRIKER MESSAGE STRUCTURE:
Use X to assign a PWM value to the strike [0, 255]
Y and Z are not used
'''

# Serial communication constants; Make sure these match device params!
DEVICE_PATH = "/dev/ttyACM0" # replace this once we know what it should be
BAUD        = 9600
TIMEOUT     = 0.1

class DemoNode(Node):

    def __init__(self, name):
        super().__init__(name)
        
        self.striker_subscriber = self.create_subscription(Float32, '/end_effector/strike', self.serial_callback, 10)
        #self.gripper_subscriber = self.create_subscription(JointState, '/gripper_states', self.gripper_callback, 10)
        
        self.ser = serial.Serial(DEVICE_PATH, BAUD, timeout = TIMEOUT)
        self.ser.reset_input_buffer()
    
    def send_data(self, toSend):
        self.get_logger().info("Sending message: " + toSend)
        self.ser.write(bytes(toSend, 'utf-8'))
        #self.ser.write(bytes("255", 'utf-8'))
        startTime = time.time()
        while time.time() - startTime < 0.1:
            pass
        self.receive_data()
        
    def receive_data(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8').rstrip()
            self.get_logger().info("Received message: " + line)
    
    def serial_callback(self, msg):
        pwmVal = int(msg.data)
        self.send_data(str(pwmVal))
        
        



def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the DEMO node.
    node = DemoNode('end_effector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()