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

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import imutils
import time
import cv2
import numpy as np

import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image

app = None

#
#  Detector Node Class
#
class DetectorNode(Node):
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

        # initialize the output frame and a lock used to ensure thread-safe
        # exchanges of the output frames (useful when multiple browsers/tabs
        # are viewing the stream)
        self.outputFrame = None
        self.lock = threading.Lock()
        # initialize a flask object
        self.app = Flask(__name__)
        # initialize the video stream and allow the camera sensor to
        # warmup
        #vs = VideoStream(usePiCamera=1).start()
        self.HZ = 24
        # vs = VideoStream(src=0, framerate=HZ).start()
        time.sleep(2.0)

        # Thresholds in Hmin/max, Smin/max, Vmin/max
        self.hsvlimits = np.array([[20, 30], [90, 170], [60, 255]])

        # Create a publisher for the processed (debugging) images.
        # Store up to three images, just in case.
        self.pubrgb = self.create_publisher(Image, name+'/image_raw', 3)
        self.pubbin = self.create_publisher(Image, name+'/binary',    3)

        # Set up the OpenCV bridge.
        self.bridge = cv_bridge.CvBridge()

        # Finally, subscribe to the incoming image topic.  Using a
        # queue size of one means only the most recent message is
        # stored for the next subscriber callback.
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process, 1)

        # Report.
        self.get_logger().info("Ball detector running...")
        t = threading.Thread(target=self.detect_motion, args=(None,))
        t.daemon = True
        t.start()
        # start the flask app
        app.run(host='0.0.0.0', port=5000, debug=False,
            threaded=True, use_reloader=False)

    # Shutdown
    def shutdown(self):
        # No particular cleanup, just shut down the node.
        self.destroy_node()


    # Process the image (detect the ball).
    def process(self, msg):
        # Confirm the encoding and report.
        assert(msg.encoding == "rgb8")
        # self.get_logger().info(
        #     "Image %dx%d, bytes/pixel %d, encoding %s" %
        #     (msg.width, msg.height, msg.step/msg.width, msg.encoding))

        # Convert into OpenCV image, using RGB 8-bit (pass-through).
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        
    @app.route("/")
    def index():
        global start
        # return the rendered template
        print(time.time() - start)
        start = time.time()
        return render_template("index.html")

    def detect_motion(self, frameCount):
        # grab global references to the video stream, output frame, and
        # lock variables
        global vs, outputFrame, lock
        while True:
            time.sleep(1 / self.HZ)
            # read the next frame from the video stream, resize it,
            # convert the frame to grayscale, and blur it
            val, frame = self.cap.read() #np.ones((400, 400)) * np.sin(time.time() - start) * 127 + 127 # vs.read()
            frame = imutils.resize(frame, width=400)
            # acquire the lock, set the output frame, and release the
            # lock
            if lock.acquire(timeout=10):
                outputFrame = frame.copy()
                lock.release()
                
    def generate():
        global outputFrame, lock
        while True:
            # wait until the lock is acquired
            if lock.acquire(timeout=10):
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if outputFrame is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                # ensure the frame was successfully encoded
                if not flag:
                    print('AHHH')
                    continue
                lock.release()
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

    @app.route("/video_feed")
    def video_feed():
        global start
        # return the response generated along with the specific media
        # type (mime type)
        # print(time.time - start)
        # if time.time() - start > 0.1:
        #     raise Exception(f'TOO LONG IN VIDEO {time.time() - start}')
        return Response(DetectorNode.generate(),
            mimetype = "multipart/x-mixed-replace; boundary=frame")


#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = DetectorNode('trial')

    # Spin the node until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
