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
from sklearn.cluster import KMeans

# ROS Imports
import rclpy
import cv_bridge
# from utils.pyutils import ros_print
from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Point, Pose, PoseArray, Vector3
from nav_msgs.msg       import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

FPS = 15.0

RGB_TOPIC = '/raw_image'
BOARD = 'Board'

STORAGE_LEN = 15

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
        self.hsv_thresh = np.array([[80, 124], [17, 77], [0, 130]])

        # publishers
        self.im_pub = self.create_publisher(Image, name + '/binary/' + BOARD, 3)
        self.im_raw_pub = self.create_publisher(Image, name + '/raw_image', 3)
        self.board_pub = self.create_publisher(Pose, name + '/pose', 3)

        # eroding and dilating
        #self.filter = lambda b: cv2.erode(cv2.dilate(b, None, iterations=2), None, iterations=3)
        self.filter = lambda b: cv2.dilate(cv2.erode(cv2.dilate(b, None, iterations=2), None, iterations=9), None, iterations=7)

        self.last_bins = []
        for i in range(STORAGE_LEN):
            self.last_bins.append(None)

        
        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # camera space to world frame transformation variables
        self.x0 = 0.323
        self.y0 = -0.002
        self.last_markerCorners = None
        self.last_markerIds = None

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(Image, '/image_raw', self.process, 1)
        


    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")

        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2
        
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

        binary = cv2.inRange(hsv, self.hsv_thresh[:,0], self.hsv_thresh[:,1])
        binary = self.filter(binary)

        # if needed, seed the storage queue with the first binary seen
        for i in range(STORAGE_LEN):
            if self.last_bins[i] is None:
                self.last_bins[i] = binary
        
        # initialize the average for scope
        avg_bin = cv2.addWeighted(binary, 1, binary, 0, 0.0)
        # merge the last N binaries to get a more reliable board edge
        for i in range(STORAGE_LEN):
            avg_bin = cv2.addWeighted(avg_bin, 1, self.last_bins[STORAGE_LEN - i - 1], 1, 0.0)

        #(contours, hierarchy) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (contours, hierarchy) = cv2.findContours(avg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.im_pub.publish(self.bridge.cv2_to_imgmsg(avg_bin, "mono8"))

        if len(contours) > 0:
            #contour = max(contours, key=cv2.contourArea)
            contours = list(contours)
            contours.sort(reverse=True, key=cv2.contourArea)
            contours = contours[:1] # select largest contour only --> the board
            #area = cv2.contourArea(contour)
            for contour in contours:
                if cv2.contourArea(contour) < 0:
                    return
                
                x,y,w,h = cv2.boundingRect(contour)
                # k=0
                # expected_area = radius**2 * np.pi

                # if not np.isclose(expected_area / area, 1.0, atol=0.65):
                #     return

                # Draw the circle (yellow) and centroid (red) on the
                # original image.
                cx, cy = x + (w)//2, y + (h)//2
                # if not (puck == TEN and k > 1):
                # frame = cv2.rectangle(frame, (x, y), (x+w, y+h), self.red, 2) 
                # cv2.drawContours(frame, [contour],0,(0,191,255),2)


                rect = cv2.minAreaRect(contour)

                x = rect[0][0] 
                box = cv2.boxPoints(rect) 
                box = np.int0(box) 
                self.get_logger().info(f'{[box]}')
                frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2) 

                # xyCenter = self.pixelToWorld(frame, cx, cy, self.x0, self.y0)
                # if xyCenter is None:
                #     # ros_print(self, 'globalization failed')
                #     return
                # (xc, yc) = xyCenter
                # pose = Pose()
                # pose.position.x = float(xc)
                # pose.position.y = float(yc)
                # pose.orientation.x = float(self.IDS[puck])
                    
        #self.pub.publish(posearray_msg)
        board_pose = Pose()
        board_pose.position.x = rect[0][0]
        board_pose.position.y = rect[0][1]
        board_pose.position.z = rect[2]
        self.board_pub.publish(board_pose)
        self.im_raw_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        self.last_bins.append(binary)
        self.last_bins = self.last_bins[1:]


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
        if True:
            cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        # Abort if not all markers are detected.
        if (markerIds is None or len(markerIds) != 4 or set(markerIds.flatten()) != set([1,2,3,0])):
            if self.last_markerCorners == None:
                return None
            markerCorners = self.last_markerCorners
            markerIds = self.last_markerIds
            # ros_print(self, 'WARNING: LOST MAPPING')
            # return None

        self.last_markerCorners = markerCorners
        self.last_markerIds = markerIds

        # Determine the center of the marker pixel coordinates.
        uvMarkers = np.zeros((4,2), dtype='float32')
        for i in range(len(markerIds)):
            uvMarkers[i,:] = np.mean(markerCorners[i], axis=1)

        # Calculate the matching World coordinates of the 4 Aruco markers.
        # DX = 0.1016
        # DY = 0.06985
        # xyMarkers = np.float32([[x0+dx, y0+dy] for (dx, dy) in
        #                         [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])
            
        for cx, cy in uvMarkers:
            cv2.circle(image, (int(cx), int(cy)), 5, self.red, -1)
        
        H = 0.1
        W = H
        xyMarkers = np.float32([[0.019 + H/2, 0.3338 + W / 2],
                                [1.12 + H/2, 0.343 + W/2],
                                [0.018 + H/2, -0.175 - W/2], 
                                [1.13 + H/2, -0.158  - W/2 - 0.0275]])
        
        xyMarkers = np.float32([xyMarkers[i] for i in markerIds])
        assert(len(xyMarkers) == len(markerIds))

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

        return xyObj[::1]

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('boarddetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
