import cv2
import mediapipe as mp
import numpy as np
import imageio
import time
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# GIF paths
tongue_gif = "assets/tongue.gif"
closed_eyes_gif = "assets/closed_eyes.gif"
monkey_gif = "assets/monkey.gif"
dance_gif = "assets/dance.gif"
finger_image = "assets/images.jpeg"

# Load GIFs
def load_gif(path):
    try:
        gif_frames = imageio.mimread(path)
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif_frames]
        return frames_bgr
    except Exception as e:
        print(f"Error loading GIF: {e}")
        return None

tongue_frames = load_gif(tongue_gif)
eyes_frames = load_gif(closed_eyes_gif)
monkey_frames = load_gif(monkey_gif)
dance_frames = load_gif(dance_gif)

# Load finger image
def load_image(path):
    try:
        img = cv2.imread(path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

finger_img = load_image(finger_image)
if finger_img is None:
    print(f"Warning: Could not load finger image from {finger_image}")
else:
    print(f"Finger image loaded successfully: {finger_img.shape}")

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    ear = (v1 + v2) / (2.0 * h)
    return ear

def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

def mouth_open_ratio(landmarks):
    # Points for upper and lower lip
    upper_lip = np.array([landmarks[13].x, landmarks[13].y])
    lower_lip = np.array([landmarks[14].x, landmarks[14].y])
    left_corner = np.array([landmarks[78].x, landmarks[78].y])
    right_corner = np.array([landmarks[308].x, landmarks[308].y])
    
    # Vertical distance (mouth height)
    mouth_height = np.linalg.norm(upper_lip - lower_lip)
    # Horizontal distance (mouth width)
    mouth_width = np.linalg.norm(left_corner - right_corner)
    
    # Ratio: if height is significant compared to width, mouth is open
    if mouth_width > 0:
        return mouth_height / mouth_width
    return 0

def is_finger_up(hand_landmarks):
    # Check if index finger is up
    # Landmark indices for index finger: 8 (tip), 6 (PIP), 5 (MCP)
    tip = hand_landmarks.landmark[8]
    pip = hand_landmarks.landmark[6]
    mcp = hand_landmarks.landmark[5]
    
    # Simple check: finger is up if tip is above both PIP and MCP joints
    # This is the most basic check - just see if the finger is extended upward
    if tip.y < pip.y and tip.y < mcp.y:
        return True
    return False

EYE_AR_THRESH = 0.20
MOUTH_AR_THRESH = 0.55
MOUTH_OPEN_THRESH = 0.15  # Threshold for detecting open mouth (teeth showing)

cap = cv2.VideoCapture(0)
frames_for_gif = []

reaction_mode = None
reaction_index = 0

print("Press Q to quit...")

while True:
    success, frame = cap.read()
    if not success:
        break
    # Flip frame horizontally to fix mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # Check for hands (works independently of face detection)
    two_hands = False
    finger_up = False
    if hand_results.multi_hand_landmarks:
        num_hands = len(hand_results.multi_hand_landmarks)
        if num_hands >= 2:
            two_hands = True
        else:
            # Check for finger up only if not two hands
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if is_finger_up(hand_landmarks):
                    finger_up = True
                    break

    # Initialize face detection variables
    eyes_closed = False
    tongue_out = False
    teeth_showing = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [263, 387, 385, 362, 380, 373]
        left_EAR = eye_aspect_ratio(landmarks, left_eye_idx)
        right_EAR = eye_aspect_ratio(landmarks, right_eye_idx)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        mar = mouth_aspect_ratio(landmarks)
        mor = mouth_open_ratio(landmarks)

        eyes_closed = avg_EAR < EYE_AR_THRESH
        tongue_out = mar > MOUTH_AR_THRESH
        teeth_showing = mor > MOUTH_OPEN_THRESH and mar < MOUTH_AR_THRESH  # Open mouth but not tongue

    # Determine reaction mode (priority: two hands > finger > face reactions)
    if two_hands:
        reaction_mode = "dance"
        cv2.putText(frame, "dance!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    elif finger_up:
        reaction_mode = "finger"
        cv2.putText(frame, "finger up", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    elif eyes_closed:
        reaction_mode = "eyes"
        cv2.putText(frame, "hell nah", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    elif tongue_out:
        reaction_mode = "tongue"
        cv2.putText(frame, "freak of nature", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    elif teeth_showing:
        reaction_mode = "monkey"
        cv2.putText(frame, "monkey", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    else:
        reaction_mode = None
        cv2.putText(frame, "Normal", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Display main camera feed
    cv2.imshow("Freak Detector", frame)

    # Reaction window
    if reaction_mode == "dance" and dance_frames:
        gif_frame = dance_frames[reaction_index % len(dance_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "finger":
        if finger_img is not None:
            # Resize image to fit window
            display_img = cv2.resize(finger_img, (300, 200))
            cv2.imshow("Reaction", display_img)
        else:
            # Fallback if image not loaded
            blank = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.putText(blank, "Image not found", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Reaction", blank)
    elif reaction_mode == "eyes" and eyes_frames:
        gif_frame = eyes_frames[reaction_index % len(eyes_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "tongue" and tongue_frames:
        gif_frame = tongue_frames[reaction_index % len(tongue_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    elif reaction_mode == "monkey" and monkey_frames:
        gif_frame = monkey_frames[reaction_index % len(monkey_frames)]
        cv2.imshow("Reaction", gif_frame)
        reaction_index += 1
    else:
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(blank, "Not Freaky", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.imshow("Reaction", blank)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
