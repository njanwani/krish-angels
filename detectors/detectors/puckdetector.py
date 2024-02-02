#!/usr/bin/env python3
#
#   balldetector.py
#
#   Detect the tennis balls with OpenCV.
#
#   Node:           /balldetector
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /balldetector/binary        Intermediate binary image
#                   /balldetector/image_raw     Debug (marked up) image
#
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge
from utils.pyutils import *

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

FPS = 15.0

RGB_TOPIC = '/raw_image'
TEN_TOPIC = '/ten'
TWENTY_TOPIC = '/twenty'
QUEEN_TOPIC = '/queen'
STRIKER_TOPIC = '/striker'

TEN = 'Ten'
TWENTY = 'Twenty'
QUEEN = 'Queen'
STRIKER = 'Striker'

#
#  Detector Node Class
#
class DetectorNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red = (255,   0,   0)
    green = (  0, 255,   0)
    blue = (  0,   0, 255)
    yellow = (255, 255,   0)
    white = (255, 255, 255)
    
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.HSV_LIMITS = {}
        self.HSV_LIMITS[TEN] = np.array([[1, 16], [110, 204], [100, 197]])
        self.HSV_LIMITS[TWENTY] = None
        self.HSV_LIMITS[QUEEN] = np.array([[100, 112], [132, 177], [43, 87]])
        self.HSV_LIMITS[STRIKER] = np.array([[83, 115], [34, 105], [0, 42]])

        # publishers
        self.PUB = {}
        self.PUB[TEN] =       self.create_publisher(MarkerArray, name + TEN_TOPIC,3)
        self.PUB[TWENTY] =    self.create_publisher(MarkerArray, name + TWENTY_TOPIC,3)
        self.PUB[QUEEN] =     self.create_publisher(MarkerArray, name + QUEEN_TOPIC,3)
        self.PUB[STRIKER] =   self.create_publisher(MarkerArray, name + STRIKER_TOPIC,3)
        
        self.IM_PUB = {}
        self.IM_PUB[RGB_TOPIC] = self.create_publisher(Image, name + RGB_TOPIC, 3)
        self.IM_PUB[TEN] = self.create_publisher(Image, name + '/binary_' + TEN, 3)
        self.IM_PUB[TWENTY] = self.create_publisher(Image, name + '/binary_' + TWENTY, 3)
        self.IM_PUB[QUEEN] = self.create_publisher(Image, name + '/binary_' + QUEEN, 3)
        self.IM_PUB[STRIKER] = self.create_publisher(Image, name + '/binary_' + STRIKER, 3)
        
        # annotation colors
        self.COLOR = {}
        self.COLOR[TEN] = self.red
        self.COLOR[TWENTY] = self.blue
        self.COLOR[QUEEN] = self.green
        self.COLOR[STRIKER] = self.white
        
        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # camera space to world frame transformation variables
        self.x0 = 0.323
        self.y0 = -0.002

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)


    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")

        id_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        hsv = cv2.cvtColor(id_frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = id_frame.shape
        uc = W//2
        vc = H//2
        
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            id_frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        
        for puck in self.HSV_LIMITS:
            if puck == TWENTY: continue
            frame = id_frame
            binary = cv2.inRange(hsv, self.HSV_LIMITS[puck][:,0], self.HSV_LIMITS[puck][:,1])
            iters = 1

            # Erode and Dilate. Definitely adjust the iterations!
            iter = 0
            binary = cv2.erode(binary, None, iterations=iter)
            binary = cv2.dilate(binary, None, iterations=iter)

            # Find contours in the mask and initialize the current
            # (x, y) center of the ball
            (contours, hierarchy) = cv2.findContours(binary,
                                                     cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)

            # Draw all contours on the original image for debugging.
            # cv2.drawContours(frame, contours, -1, self.COLOR[puck], 2)

            # Only proceed if at least one contour was found.  You may
            # also want to loop over the contours_puck...
            self.IM_PUB[puck].publish(self.bridge.cv2_to_imgmsg(binary, "mono8"))
            if len(contours) > 0:
                # Pick the largest contour.
                poses = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 150:
                        continue
                    
                    ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
                    x,y,w,h = cv2.boundingRect(contour) 
                    ur     = int(ur)
                    vr     = int(vr)
                    radius = int(radius)
                    
                    # expected_area = radius**2 * np.pi
                    # if not np.isclose(expected_area / area, 1.0, atol=0.5):
                    #     continue
                    
                    if puck == STRIKER:
                        discard = False
                        for marker in markerCorners:
                            if np.isclose(np.linalg.norm([x + w//2, y + h//2] - np.mean(marker, axis=1)), 0, atol=20):
                                discard = True
                                continue
                        
                        if discard:
                            continue

                    # Draw the circle (yellow) and centroid (red) on the
                    # original image.
                    cx, cy = x + (w)//2, y + (h)//2
                    cv2.ellipse(frame, (cx, cy), (w//2, h//2), 0, 0, 360, self.COLOR[puck],  2)
                    cv2.circle(frame, (cx, cy), 5, self.COLOR[puck], -1)

                    xyCenter = self.pixelToWorld(frame, cx, cy, self.x0, self.y0)
                    if xyCenter is None:
                        ros_print(self, "Unable to execute mapping for circle")
                    else:
                        (xc, yc) = xyCenter
                        pose = Pose()
                        pose.position.x = float(xc)
                        pose.position.y = float(yc)
                        pose.position.z = 0.02
                        # poses.append(pose)
                        
                        msg = Marker()
                        msg.id = len(poses)
                        msg.action = 0
                        color = ColorRGBA()
                        color.r, color.g, color.b = list(np.array(self.COLOR[puck]) / 255)
                        color.a = 0.9
                        msg.color = color 
                        msg.header.frame_id = 'world'
                        msg.type = 3
                        msg.pose = pose
                        scale = Vector3()
                        scale.x = 0.05
                        scale.y = 0.05
                        scale.z = 0.05
                        msg.scale = scale
                        # msg.lifetime = 0
                        dur = rclpy.duration.Duration(seconds=0)
                        msg.lifetime = dur.to_msg()
                        msg.frame_locked = True
                        
                        poses.append(msg)
                        
                
                # msg = PoseArray()
                # msg.poses = poses
                # msg.header.frame_id = 'world'
                # self.PUB[puck].publish(msg)
                
                msg = MarkerArray()
                msg.markers = poses
                self.PUB[puck].publish(msg)
                
                
                
                
    

        # # Convert the frame back into a ROS image and republish.
        # frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
        # frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)
        self.IM_PUB[RGB_TOPIC].publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))


# Pixel Conversion
    def pixelToWorld(self, image, u, v, x0, y0, annotateImage=False):
        '''
        Convert the (u,v) pixel position into (x,y) world coordinates
        Inputs:
          image: The image as seen by the camera
          u:     The horizontal (column) pixel coordinate
          v:     The vertical (row) pixel coordinate
          x0:    The x world coordinate in the center of the marker paper
          y0:    The y world coordinate in the center of the marker paper
          annotateImage: Annotate the image with the marker information

        Outputs:
          point: The (x,y) world coordinates matching (u,v), or None

        Return None for the point if not all the Aruco markers are detected
        '''

        # Detect the Aruco markers (using the 4X4 dictionary).
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        if annotateImage:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        # Abort if not all markers are detected.
        if (markerIds is None or len(markerIds) != 4 or
            set(markerIds.flatten()) != set([1,2,3,4])):
            return None


        # Determine the center of the marker pixel coordinates.
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(4):
            uvMarkers[markerIds[i]-1,:] = np.mean(markerCorners[i], axis=1)

        # Calculate the matching World coordinates of the 4 Aruco markers.
        DX = 0.1016
        DY = 0.06985
        xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
                                [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])


        # Create the perspective transform.
        M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)

        # Map the object in question.
        uvObj = np.float32([u, v])
        xyObj = cv2.perspectiveTransform(uvObj.reshape(1,1,2), M).reshape(2)


        # Mark the detected coordinates.
        if annotateImage:
            # cv2.circle(image, (u, v), 5, (0, 0, 0), -1)
            s = "(%7.4f, %7.4f)" % (xyObj[0], xyObj[1])
            cv2.putText(image, s, (u-80, v-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2, cv2.LINE_AA)

        return xyObj

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('puckdetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
