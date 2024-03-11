# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='/home/robot/robotws/src/detectors/detectors/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

images = []
results = []
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mp_image)

    # STEP 5: Process the result. In this case, visualize it.
    top_gesture = None
    if recognition_result.gestures != []:
        top_gesture = recognition_result.gestures[0][0].category_name
   
    if recognition_result.hand_landmarks:
        for handLms in recognition_result.hand_landmarks:
            for id, lm in enumerate(handLms):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if top_gesture is not None: print(top_gesture)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
  # STEP 3: Load the input image.
#   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   image = mp.Image.create_from_file(image_file_name)


display_batch_of_images_with_gestures_and_hand_landmarks(images, results)