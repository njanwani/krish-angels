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
TEN_TOPIC = '/ten'
TWENTY_TOPIC = '/twenty'
QUEEN_TOPIC = '/queen'
STRIKER_TOPIC = '/striker'
FLIPPED_TOPIC = '/flipped'

TEN = 'Ten'
TWENTY = 'Twenty'
QUEEN = 'Queen'
STRIKER = 'Striker'
BOARD = 'Board'
FLIPPED = 'Flipped'

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
    pink = (255, 192, 203)
    
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Detect flipped within board limits
        # self.board_sub = self.create_subscription(Pose, '/boarddetector/pose', self.board_cb, 1)
        # self.board_corners_sub = self.create_subscription(PoseArray, '/boarddetector/board_corners', self.board_corners_cb, 1)

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.HSV_LIMITS = {}
        self.HSV_LIMITS[TEN] = np.array([[72, 93], [32, 151], [39, 156]])
        self.HSV_LIMITS[STRIKER] = np.array([[29, 44], [31, 84], [124, 192]])
        self.HSV_LIMITS[QUEEN] = np.array([[88, 117], [98, 255], [126, 199]])
        self.HSV_LIMITS[TWENTY] = np.array([[11, 18], [83, 236], [119, 194]])
        self.HSV_LIMITS[FLIPPED] = np.array([[0, 179], [0, 99], [0, 106]])

        self.hsv_board = np.array([[10, 22], [99, 224], [130, 214]])

        # publishers
        self.pub = self.create_publisher(PoseArray, '/' + name + '/pucks', 3)
        
        self.IM_PUB = {}
        self.IM_PUB[RGB_TOPIC] = self.create_publisher(Image, name + RGB_TOPIC, 3)
        self.IM_PUB[TEN] = self.create_publisher(Image, name + '/binary_' + TEN, 3)
        self.IM_PUB[TWENTY] = self.create_publisher(Image, name + '/binary_' + TWENTY, 3)
        self.IM_PUB[QUEEN] = self.create_publisher(Image, name + '/binary_' + QUEEN, 3)
        self.IM_PUB[STRIKER] = self.create_publisher(Image, name + '/binary_' + STRIKER, 3)
        self.IM_PUB[BOARD] = self.create_publisher(Image, name + '/binary_' + BOARD, 3)
        self.IM_PUB[FLIPPED] = self.create_publisher(Image, name + '/binary/' + FLIPPED, 3)

        # eroding and dilating
        self.BINARY_FILTER = {}
        self.BINARY_FILTER[TEN] = lambda b: cv2.erode(cv2.dilate(b, None, iterations=1), None, iterations=1)
        self.BINARY_FILTER[TWENTY] = lambda b: cv2.dilate(cv2.erode(b, None, iterations=1), None, iterations=1)
        self.BINARY_FILTER[QUEEN] = lambda b: cv2.dilate(cv2.erode(b, None, iterations=0), None, iterations=1)
        self.BINARY_FILTER[STRIKER] = lambda b: cv2.dilate(cv2.erode(b, None, iterations=0), None, iterations=1)
        self.BINARY_FILTER[BOARD] = lambda b: cv2.dilate(cv2.erode(b, None, iterations=2), None, iterations=1)
        self.BINARY_FILTER[FLIPPED] = lambda b: cv2.dilate(cv2.erode(b, None, iterations=2), None, iterations=1)
        # annotation colors
        self.COLOR = {}
        self.COLOR[TEN] = self.red
        self.COLOR[TWENTY] = self.blue
        self.COLOR[QUEEN] = self.green
        self.COLOR[STRIKER] = self.white

        self.IDS = {}
        self.IDS[TEN] = 0
        self.IDS[TWENTY] = 1
        self.IDS[QUEEN] = 2
        self.IDS[STRIKER] = 3
        self.IDS[FLIPPED] = 4

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
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        # ros_print(self, 'here!')


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
        
        frame = id_frame
        
        posearray_msg = PoseArray()
        poses = []
        
        for puck in self.HSV_LIMITS:
            frame = id_frame
            binary = cv2.inRange(hsv, self.HSV_LIMITS[puck][:,0], self.HSV_LIMITS[puck][:,1])
            binary = self.BINARY_FILTER[puck](binary)

            if puck == FLIPPED:
                (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                (contours, hierarchy) = cv2.findContours(binary,
                                                        cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            self.IM_PUB[puck].publish(self.bridge.cv2_to_imgmsg(binary, "mono8"))
            if puck == FLIPPED:
                if len(contours) > 0:
                    areas = [cv2.contourArea(contour) for contour in contours]
                    idx_max = np.argmax(areas)

                    board = contours[idx_max]
                    if cv2.contourArea(board) < 0:
                        self.die(frame)

                    # rect = cv2.minAreaRect(board)
                    # (x,y), (bw,bh), theta = rect
                    # box = cv2.boxPoints(rect) 
                    # box = np.int0(box) 
                    # frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

                    # area = cv2.contourArea(board)
                    # area_calc = bw*bh
                    # accuracy = area / area_calc

                    for contour in contours:
                        board = board[:,0,:].reshape((-1,1,2))
                        dist = cv2.pointPolygonTest(board[:,0,:].astype(int), contour[0][0].astype(float), False)

                        area = cv2.contourArea(contour)
                        if area <= 0:
                            continue
                        perimeter = cv2.arcLength(contour, True)
                        r = perimeter / (2 * np.pi)
                        accuracy = np.pi * r**2 / area
                        if dist > 0.0 and np.isclose(accuracy, 1.0, atol=0.7) and np.isclose(area, 100, atol=25):
                            cv2.drawContours(frame, [contour], -1,(0,191,255),2)
                            
                            ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
                            x,y,w,h = cv2.boundingRect(contour) 
                            ur     = int(ur)
                            vr     = int(vr)
                            radius = int(radius)

                            xyCenter = self.pixelToWorld(frame, ur, vr, None, None, annotateImage=False)
                            if xyCenter is None:
                                break
                            (xc, yc) = xyCenter
                            pose = Pose()
                            pose.position.x = float(xc)
                            pose.position.y = float(yc)
                            pose.orientation.x = float(self.IDS[puck])
                            poses.append(pose)
            else:
                if len(contours) > 0:
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < 50:
                            continue
                        
                        ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
                        x,y,w,h = cv2.boundingRect(contour) 
                        ur     = int(ur)
                        vr     = int(vr)
                        radius = int(radius)
                        
                        k=0
                        expected_area = radius**2 * np.pi

                        if not np.isclose(expected_area / area, 1.0, atol=0.65):
                            continue
                        
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
                        # if not (puck == TEN and k > 1):
                        cv2.ellipse(frame, (cx, cy), (w//2, h//2), 0, 0, 360, self.COLOR[puck],  2)
                        cv2.circle(frame, (cx, cy), 2, self.COLOR[puck], -1)

                        xyCenter = self.pixelToWorld(frame, cx, cy, self.x0, self.y0)
                        if xyCenter is None:
                            # ros_print(self, 'globalization failed')
                            continue
                        (xc, yc) = xyCenter
                        pose = Pose()
                        pose.position.x = float(xc)
                        pose.position.y = float(yc)
                        pose.orientation.x = float(self.IDS[puck])
                        poses.append(pose)
                
                
                
        posearray_msg.poses = poses

        self.pub.publish(posearray_msg)
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
    node = DetectorNode('puckdetector')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
