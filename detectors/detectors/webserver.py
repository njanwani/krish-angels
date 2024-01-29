#!/usr/bin/env python3

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import imutils
import time
import cv2
import numpy as np

# ROS Imports
import rclpy
import cv_bridge

from rclpy.node         import Node
from sensor_msgs.msg    import Image

app = Flask(__name__)
node = None

#
#  Detector Node Class
#
class WebNode(Node):
    # Pick some colors, assuming RGB8 encoding.
    red    = (255,   0,   0)
    green  = (  0, 255,   0)
    blue   = (  0,   0, 255)
    yellow = (255, 255,   0)
    white  = (255, 255, 255)

    # Initialization.
    def __init__(self, name, app):
        super().__init__(name)
        self.outputFrame = None
        self.app = app
        self.HZ = 24
        self.bridge = cv_bridge.CvBridge()
        self.lock = threading.Lock()
        self.sub_im1 = self.create_subscription(
            Image, 'balldetector/image_raw', lambda msg: self.cb_image(msg, 0), 1)
        self.sub_im2 = self.create_subscription(
            Image, 'balldetector/binary_circle', lambda msg: self.cb_image(msg, 1), 1)
        self.sub_im3 = self.create_subscription(
            Image, 'balldetector/binary_rectangle', lambda msg: self.cb_image(msg, 2), 1)
        self.frames = [None]*3
        # Report.
        self.get_logger().info("Webserver running...")
        

    # Shutdown
    def shutdown(self):
        self.destroy_node()


    # Process the image (detect the ball).
    def cb_image(self, msg, num):
        # assert(msg.encoding == "mono8")
        # assert(msg.encoding == "rgb8")

        # frame = self.bridge.imgmsg_to_cv2(msg, "mono8")
        frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        if msg.encoding == 'rgb8':
            frame = frame[:,:,2::-1]
        resize = 500
        frame = imutils.resize(frame, width=resize)
        with self.lock:
            self.frames[num] = frame.copy()
        
    @app.route("/")
    def index():
        # return the rendered template
        return render_template("index.html")
                
    def generate(self, num):
        while True:
            time.sleep(1 / self.HZ)
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if self.frames[num] is None:
                continue
            # encode the frame in JPEG format
            with self.lock:
                (flag, encodedImage) = cv2.imencode(".jpg", self.frames[num])
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

    @app.route("/feed1")
    def feed1():
        # return the response generated along with the specific media
        # type (mime type)
        return Response(node.generate(0),
            mimetype = "multipart/x-mixed-replace; boundary=frame")
        
    @app.route("/feed2")
    def feed2():
        # return the response generated along with the specific media
        # type (mime type)
        return Response(node.generate(1),
            mimetype = "multipart/x-mixed-replace; boundary=frame")
        
    @app.route("/feed3")
    def feed3():
        # return the response generated along with the specific media
        # type (mime type)
        return Response(node.generate(2),
            mimetype = "multipart/x-mixed-replace; boundary=frame")


def start_app(_):
    global app
    app.run(host='0.0.0.0', port=5000, debug=False,
            threaded=True, use_reloader=False)

#
#   Main Code
#
def main(args=None):
    global node
    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the detector node.
    node = WebNode('webserver', app)
    t = threading.Thread(target=start_app, args=(None,))
    t.daemon = True
    t.start()
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
