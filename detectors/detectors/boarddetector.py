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
import time
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

def ros_print(node, msg: str):
    """
    Easy print for ROS nodes
    """
    node.get_logger().info(str(msg))

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
        # self.hsv_thresh = np.array([[96, 124], [18, 57], [33, 101]]) # border
        self.hsv_thresh = np.array([[10, 28], [21, 71], [0, 206]])

        # publishers
        self.im_pub = self.create_publisher(Image, name + '/binary/' + BOARD, 3)
        self.im_raw_pub = self.create_publisher(Image, name + '/raw_image', 3)
        self.board_pub = self.create_publisher(Pose, name + '/pose', 3)
        self.board_corner_pub = self.create_publisher(PoseArray, name + '/board_corners', 3)
        self.shot_axis_pub = self.create_publisher(PoseArray, name + '/shot_axis', 3)

        # eroding and dilating
        #self.filter = lambda b: cv2.erode(cv2.dilate(b, None, iterations=2), None, iterations=3)
        self.filter = lambda b: cv2.dilate(cv2.erode(cv2.dilate(b, None, iterations=2), None, iterations=9), None, iterations=7)
        self.filter2 = lambda b: cv2.dilate(cv2.erode(b, None, iterations=0), None, iterations=0)

        self.last_bins = []
        for i in range(STORAGE_LEN):
            self.last_bins.append(None)
        self.last = time.time()
        
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
        if time.time() - self.last <= 1.0:
            return
        self.last = time.time()
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
        posearray_msg = PoseArray()
        poses = []
        # poses_coords = []

        binary = cv2.inRange(hsv, self.hsv_thresh[:,0], self.hsv_thresh[:,1])
        binary = self.filter2(binary)

        
        (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.im_pub.publish(self.bridge.cv2_to_imgmsg(binary, "mono8"))
        accuracy = 0
        if len(contours) > 0:
            #contour = max(contours, key=cv2.contourArea)
            accs = [self.get_accuracy(contour) for contour in contours]
            idx = np.argmax(accs)
            # ros_print(self, accs[idx])
            contour = contours[idx]
            # contours.sort(reverse=True, key=self.get_accuracy)
            #area = cv2.contourArea(contour)
            if cv2.contourArea(contour) < 0:
                self.die(frame)

            rect = cv2.minAreaRect(contour)
            (x,y), (bw,bh), theta = rect
            box = cv2.boxPoints(rect) 
            box = np.int0(box) 
            frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            x_sort = box.tolist()
            x_sort.sort(key=lambda x: x[0])
            x_small = x_sort[:2]
            x_large = x_sort[2:]
            x_small.sort(key=lambda x: x[1])
            x_large.sort(key=lambda x: x[1])
            # ros_print(self, x_small)
            box_sorted = [None] * 4
            box_sorted[0] = x_small[0]
            box_sorted[1] = x_small[1]
            box_sorted[2] = x_large[0]
            box_sorted[3] = x_large[1]
            _sort = list(box).sort(key=lambda x: x[0])
            for idx, pt in enumerate(box_sorted):
                cv2.putText(frame, str(idx), pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2, cv2.LINE_AA)
            theta = np.arctan2(box_sorted[1][1] - box_sorted[0][1], box_sorted[1][0] - box_sorted[0][0])
            
            w, h = (95, 70)
            p1 = np.array([w * np.cos(theta), w * np.sin(theta)])
            p2 = np.array([h * np.cos(theta - np.pi / 2), h * np.sin(theta - np.pi / 2)])
            v1 = tuple((np.array(box_sorted[0]) + p1 + p2).astype(int))
            cv2.circle(frame, v1, 5, (255, 0, 255), -1)

            p1 = np.array([-w * np.cos(theta),- w * np.sin(theta)])
            p2 = np.array([h * np.cos(theta - np.pi / 2), h * np.sin(theta - np.pi / 2)])
            v2 = tuple((np.array(box_sorted[1]) + p1 + p2).astype(int))
            cv2.circle(frame, v2, 5, (255, 0, 255), -1)

            xyShot1 = self.pixelToWorld(frame, v1[0], v1[1], self.x0, self.y0)
            if xyShot1 is None:
                self.die(frame)
                return
            shot1 = Pose()
            shot1.position.x = float(xyShot1[0])
            shot1.position.y = float(xyShot1[1])

            xyShot2 = self.pixelToWorld(frame, v2[0], v2[1], self.x0, self.y0)
            if xyShot2 is None:
                self.die(frame)
                return
            shot2 = Pose()
            shot2.position.x = float(xyShot2[0])
            shot2.position.y = float(xyShot2[1])

            shots = PoseArray()
            shots.poses = [shot1, shot2]
            self.shot_axis_pub.publish(shots)

            area = cv2.contourArea(contour)
            # ros_print(self, area)
            area_calc = bw*bh
            accuracy = area / area_calc
            # perimeter = cv2.arcLength(contour, True)  # Perimeter of first contour
            # ros_print(self, accuracy)
        
        if accuracy > 0.07:
            xyCenter = self.pixelToWorld(frame, rect[0][0], rect[0][1], self.x0, self.y0)
            self.im_raw_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
            if xyCenter is None:
                self.die(frame)
                return
            (xc, yc) = xyCenter
            board_pose = Pose()
            # board_pose.position.x = rect[0][0]
            # board_pose.position.y = rect[0][1]
            board_pose.position.x = float(xc)
            board_pose.position.y = float(yc)
            board_pose.position.z = 0.0
            self.board_pub.publish(board_pose)

            for i in range(len(box_sorted)):
                cx = box_sorted[i][0]
                cy = box_sorted[i][1]
                xyCorner = self.pixelToWorld(frame, cx, cy, self.x0, self.y0)
                if xyCorner is None:
                    self.die(frame)
                    return
                (xco, yco) = xyCorner
                pose = Pose()
                pose.position.x = float(xco)
                pose.position.y = float(yco)
                poses.append(pose)
            posearray_msg.poses = poses
            self.board_corner_pub.publish(posearray_msg)
            self.last_bins.append(binary)
            self.last_bins = self.last_bins[1:]
            

    def get_accuracy(self, contour):
        rect = cv2.minAreaRect(contour)
        (x,y), (bw,bh), theta = rect
        if bw == 0 or bh == 0:
            return -np.inf
        area = cv2.contourArea(contour)
        if area < 4000:
            return -np.inf
        area_calc = bw*bh
        accuracy = area / area_calc
        # ros_print(self, accuracy)
        return -np.abs(1 - accuracy)

    def die(self, frame):
        # self.im_raw_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))
        pass


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
