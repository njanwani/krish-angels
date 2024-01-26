#!/usr/bin/env python3
#
#   mapping.py
#
#   Demostrate how to map pixel coordinates into world coordinates.
#
#   Node:           /mapper
#   Subscribers:    /usb_cam/image_raw          Source image
#   Publishers:     /mapper/image_raw           Debug (marked up) image
#
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image
from geometry_msgs.msg import Pose, Point

x0 = 0.321
y0 = 0.0
#
#  Demo Node Class
#
class DemoNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.hsvlimits_puck = np.array([[5, 20], [150, 255], [100, 200]])
        self.hsvlimits_paper = np.array([[0, 15], [59, 127], [125, 172]])
        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)
        self.pubrec = self.create_publisher(Pose, name+'/rectangle',    3)
        self.pubcirc = self.create_publisher(Point, name+'/circle',    3)
        
        self.circ_msg = Point()
        self.rec_msg = Pose()
        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)
        
        self.circ_sub = self.create_subscription(
            Image, '/balldetector/circle', self.process, 1)

        # Report.
        self.get_logger().info("Mapper running...")

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Pixel Conversion
    def pixelToWorld(self, image, u, v, x0, y0, annotateImage=True):
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


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Cheat: swap red/blue

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Help to determine the HSV range...
        if True:
            # Draw the center lines.  Note the row is the first dimension.
            frame = cv2.line(frame, (uc,0), (uc,H-1), self.white, 1)
            frame = cv2.line(frame, (0,vc), (W-1,vc), self.white, 1)

            # Report the center HSV values.  Note the row comes first.
            self.get_logger().info(
                "HSV = (%3d, %3d, %3d)" % tuple(hsv[vc, uc]))

        
        # Threshold in Hmin/max, Smin/max, Vmin/max
        binary_puck = cv2.inRange(hsv, self.hsvlimits_puck[:,0], self.hsvlimits_puck[:,1])
        binary_paper = cv2.inRange(hsv, self.hsvlimits_paper[:,0], self.hsvlimits_paper[:,1])

        # Erode and Dilate. Definitely adjust the iterations!
        iter = 1
        binary_puck = cv2.erode(binary_puck, None, iterations=iter)
        binary_puck = cv2.dilate(binary_puck, None, iterations=iter)

        binary_paper = cv2.erode(binary_paper, None, iterations=iter)
        binary_paper = cv2.dilate(binary_paper, None, iterations=iter)

        # Find contours in the mask and initialize the current
        # (x, y) center of the ball
        (contours_puck, hierarchy) = cv2.findContours(
            binary_puck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        (contours_paper, hierarchy) = cv2.findContours(
            binary_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours on the original image for debugging.
        # cv2.drawContours(frame, contours_puck, -1, self.blue, 2)
        # cv2.drawContours(frame, contours_paper, -1, self.red, 2)

        # Only proceed if at least one contour was found.  You may
        # also want to loop over the contours_puck...
        if len(contours_puck) > 0:
            # Pick the largest contour.
            contour = max(contours_puck, key=cv2.contourArea)
            # Find the enclosing circle (convert to pixel values)
            ((ur, vr), radius) = cv2.minEnclosingCircle(contour)
            x,y,w,h = cv2.boundingRect(contour) 
            ur     = int(ur)
            vr     = int(vr)
            radius = int(radius)

            # Draw the circle (yellow) and centroid (red) on the
            # original image.
            # cv2.circle(frame, (ur, vr), int(radius), self.yellow,  2)
            # cv2.circle(frame, (ur, vr), 5,           self.red,    -1)
            
            if cv2.contourArea(contour) > 500:
                cx, cy = x + (w)//2, y + (h)//2
                center = self.pixelToWorld(frame, cx, cy, x0, y0)
                cv2.ellipse(frame, (cx, cy), (w//2, h//2), 0, 0, 360, self.yellow,  2)
                cv2.circle(frame, (cx, cy), 5,           self.red,    -1)
                if center is not None:
                    gx, gy = center
                    if np.linalg.norm(np.array([self.circ_msg.x - gx, self.circ_msg.y - gy])) > 0.2:
                        self.c_found = False
                        
                    if not self.c_found:
                        self.circ_msg.x = gx
                        self.circ_msg.y = gy
                        self.circ_msg.z = 0
                        self.pubcirc.publish(self.msg)
                        self.c_found = True
                        
                    # Report.
                    self.get_logger().info(
                        "Found Ball enclosed by radius %d about (%d,%d)" %
                        (radius, ur, vr))
            
        if len(contours_paper) > 0:
            # Pick the largest contour.
            contour = max(contours_paper, key=cv2.contourArea)

            # Find the enclosing circle (convert to pixel values)
            x,y,w,h = cv2.boundingRect(contour) 
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            if cv2.contourArea(contour) > 2000:
                cv2.drawContours(frame,[box],0,(0,191,255),2)
                cv2.circle(frame, (x+w//2, y+h//2), 5, self.red, -1)

                # Report.
                self.get_logger().info(f"Found Paper enclosed by {(x,y)} and {(y+w, x+h)}")

        # Convert the frame back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

        # Also publish the binary_puck (black/white) image.
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary_puck))
        self.pubbin.publish(self.bridge.cv2_to_imgmsg(binary_paper))
        
        
        

        # Grab the image shape, determine the center pixel.
        (H, W, D) = frame.shape
        uc = W//2
        vc = H//2

        # Assume the center of marker sheet is at the world origin.
        

        # Convert the center of the image into world coordinates.
        


        # Convert the image back into a ROS image and republish.
        self.pubrgb.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DemoNode('mapper')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
