import cv2
import mediapipe as mp
import math
import time
import winsound
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. Custom Voice Alarm Setup
# ==========================================

# Find alarm.wav in the same folder as main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALARM_SOUND = os.path.join(
    BASE_DIR,
    "bishwash.wav"
)


def speak_alarm():

    print("Playing custom voice alarm...")

    # Check whether alarm.wav exists
    if not os.path.exists(ALARM_SOUND):

        print("ERROR: alarm.wav not found!")
        print("Expected location:")
        print(ALARM_SOUND)

        return

    try:

        winsound.PlaySound(
            ALARM_SOUND,
            winsound.SND_FILENAME
        )

        print("Custom voice alarm completed.")

    except Exception as e:

        print("VOICE ERROR:")
        print(type(e).__name__)
        print(e)


# ==========================================
# 2. Load Face Landmarker Model
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)


landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ==========================================
# 3. Open Webcam
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Camera started.")
print("Press Q to quit.")


# ==========================================
# 4. Eye Landmark Indices
# ==========================================

LEFT_EYE = [
    33,
    133,
    159,
    145,
    158,
    144
]


RIGHT_EYE = [
    362,
    263,
    386,
    374,
    385,
    380
]


# ==========================================
# 5. Drowsiness Settings
# ==========================================

# Based on your testing:
#
# Open eye  ≈ 0.44
# Closed eye ≈ 0.19

EAR_THRESHOLD = 0.25


# Eyes must remain closed
# for this many seconds

DROWSINESS_TIME = 2.0


# ==========================================
# 6. Distance Function
# ==========================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ==========================================
# 7. Calculate EAR
# ==========================================

def calculate_ear(landmarks, eye_indices):

    # Horizontal distance

    horizontal = distance(
        landmarks[eye_indices[0]],
        landmarks[eye_indices[1]]
    )


    # Vertical distance 1

    vertical1 = distance(
        landmarks[eye_indices[2]],
        landmarks[eye_indices[3]]
    )


    # Vertical distance 2

    vertical2 = distance(
        landmarks[eye_indices[4]],
        landmarks[eye_indices[5]]
    )


    # Eye Aspect Ratio

    ear = (
        vertical1 + vertical2
    ) / (2.0 * horizontal)


    return ear


# ==========================================
# 8. Timer Variables
# ==========================================

eyes_closed_start = None

alarm_triggered = False


# ==========================================
# 9. Main Processing Loop
# ==========================================

timestamp = 0


while True:

    # --------------------------------------
    # Capture frame
    # --------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Could not read frame.")
        break


    # --------------------------------------
    # Mirror camera
    # --------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------
    # Convert BGR → RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Create MediaPipe image
    # --------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------
    # Face detection
    # --------------------------------------

    results = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    timestamp += 1


    # ======================================
    # FACE DETECTED
    # ======================================

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]


        # ==================================
        # Calculate Left EAR
        # ==================================

        left_ear = calculate_ear(
            face_landmarks,
            LEFT_EYE
        )


        # ==================================
        # Calculate Right EAR
        # ==================================

        right_ear = calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )


        # ==================================
        # Average EAR
        # ==================================

        average_ear = (
            left_ear + right_ear
        ) / 2.0


        # ==================================
        # Eye State
        # ==================================

        if average_ear < EAR_THRESHOLD:

            eye_status = "EYES CLOSED"


            # Start timer

            if eyes_closed_start is None:

                eyes_closed_start = time.time()


            # Calculate how long eyes
            # have remained closed

            closed_duration = (
                time.time()
                - eyes_closed_start
            )


        else:

            eye_status = "EYES OPEN"


            # Reset timer

            eyes_closed_start = None

            closed_duration = 0


            # Allow another alarm
            # after eyes open again

            alarm_triggered = False


        # ==================================
        # Drowsiness Detection
        # ==================================

        drowsiness_detected = (
            closed_duration >= DROWSINESS_TIME
        )


        # ==================================
        # Voice Alarm
        # ==================================

        if drowsiness_detected:

            if not alarm_triggered:

                print()
                print("==============================")
                print("DROWSINESS DETECTED!")
                print("==============================")


                # Play your custom recording

                speak_alarm()


                # Prevent alarm from
                # repeatedly triggering
                # during the same event

                alarm_triggered = True


        # ==================================
        # Display EAR
        # ==================================

        cv2.putText(
            frame,
            f"Left EAR: {left_ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Right EAR: {right_ear:.3f}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Average EAR: {average_ear:.3f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==================================
        # Eye Status
        # ==================================

        cv2.putText(
            frame,
            f"Status: {eye_status}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ==================================
        # Closed Duration
        # ==================================

        cv2.putText(
            frame,
            f"Closed: {closed_duration:.1f} sec",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # ==================================
        # Drowsiness Status
        # ==================================

        if drowsiness_detected:

            status_text = "DROWSINESS DETECTED!"

        else:

            status_text = "NORMAL"


        cv2.putText(
            frame,
            status_text,
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        # ==================================
        # Draw Eye Landmarks
        # ==================================

        height, width, _ = frame.shape


        # ----------------------------------
        # Left eye
        # ----------------------------------

        for index in LEFT_EYE:

            landmark = face_landmarks[index]


            x = int(
                landmark.x * width
            )


            y = int(
                landmark.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ----------------------------------
        # Right eye
        # ----------------------------------

        for index in RIGHT_EYE:

            landmark = face_landmarks[index]


            x = int(
                landmark.x * width
            )


            y = int(
                landmark.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


    # ======================================
    # NO FACE DETECTED
    # ======================================

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        # Reset everything

        eyes_closed_start = None

        alarm_triggered = False


    # ======================================
    # Display Camera
    # ======================================

    cv2.imshow(
        "Drowsiness Detector",
        frame
    )


    # ======================================
    # Q = Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 10. Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print("Program closed.")