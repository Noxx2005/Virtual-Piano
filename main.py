import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyglet
import time
import urllib.request
import os

# Constants
wCam, hCam = 640, 480
w, h = 40, 150
playlist = ['./tones/tone1.mp3', './tones/tone2.mp3', './tones/tone3.mp3', './tones/tone4.mp3', './tones/tone5.mp3', './tones/tone6.mp3', './tones/tone7.mp3']
handpoints = [(i * 50 + 80, 130) for i in range(7)]
ractpoints = [(i * 50 + 60, 0) for i in range(7)]

# MediaPipe Tasks hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# Download the hand landmarker model if not present
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print('Downloading hand landmarker model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize MediaPipe HandLandmarker (Tasks API)
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# Pre-load all tones as StaticSource so they can be replayed without re-reading disk
songs = [pyglet.media.StaticSource(pyglet.media.load(path)) for path in playlist]


def findHands(img, detection_result, draw=True):
    """Draws hand landmarks on the image from a HandLandmarker detection result."""
    if draw and detection_result.hand_landmarks:
        hi, wi, _ = img.shape
        for hand_landmarks in detection_result.hand_landmarks:
            pts = [(int(lm.x * wi), int(lm.y * hi)) for lm in hand_landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(img, pts[start], pts[end], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(img, pt, 5, (255, 0, 255), cv2.FILLED)
    return img


def findPositions(img, detection_result, draw=True):
    """Finds the pixel positions of landmarks on detected hands."""
    lmList = []
    hi, wi, _ = img.shape

    for hand_landmarks in detection_result.hand_landmarks:
        xList, yList, lList = [], [], []
        for id, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * wi), int(lm.y * hi)
            xList.append(cx)
            yList.append(cy)
            lList.append([id, cx, cy])

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        lmList.append(lList)

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                          (0, 255, 0), 2)

    return img, lmList


def playMusic(img, p1, p2):
    """Plays music based on the hand position on the virtual piano."""
    for i, (hp, rp) in enumerate(zip(handpoints, ractpoints)):
        if (hp[0] - 7 < p1 < hp[0] + 7) and (hp[1] - 7 < p2 < hp[1] + 7):
            cv2.rectangle(img, rp, (rp[0] + w, rp[1] + h), (255, 0, 255), -1)
            player = pyglet.media.Player()
            player.queue(songs[i])
            player.play()
            pyglet.app.platform_event_loop.start()
            pyglet.clock.tick()
            pyglet.app.platform_event_loop.stop()
            time.sleep(0.1)
            break


while True:
    success, img = cap.read()
    if not success:
        continue

    # Convert to MediaPipe Image and run detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    img = findHands(img, detection_result)
    img, lmlist = findPositions(img, detection_result)

    # Draw rectangles representing piano keys on the image
    for rect, point in zip(ractpoints, handpoints):
        cv2.rectangle(img, rect, (rect[0] + w, rect[1] + h), (255, 0, 255), 2)
        cv2.circle(img, point, 7, (255, 0, 255), cv2.FILLED)

    # Check if hand landmarks are detected and play music accordingly
    for hand in lmlist[:2]:
        p1, p2 = hand[8][1:]
        p3, p4 = hand[12][1:]
        playMusic(img, p1, p2)
        playMusic(img, p3, p4)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
