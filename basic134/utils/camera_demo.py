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

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
HZ = 24
# vs = VideoStream(src=0, framerate=HZ).start()
cap = cv2.VideoCapture(0)
time.sleep(2.0)

start = time.time()
@app.route("/")
def index():
    global start
    # return the rendered template
    print(time.time() - start)
    start = time.time()
    return render_template("index.html")

def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock
    while True:
        time.sleep(1 / HZ)
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        val, frame = cap.read() #np.ones((400, 400)) * np.sin(time.time() - start) * 127 + 127 # vs.read()
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
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	t = threading.Thread(target=detect_motion, args=(None,))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host='0.0.0.0', port=5000, debug=False,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
