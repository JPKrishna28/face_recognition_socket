import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from skimage.feature import local_binary_pattern

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Function to get landmarks
def get_landmarks(gray, rect):
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks


# Function to check for eye blinks
def detect_blink(landmarks):
    if landmarks is None or len(landmarks) < 68:
        return False

    # Get left and right eye landmarks
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]

    # Calculate the eye aspect ratios
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average the eye aspect ratio
    ear = (left_ear + right_ear) / 2.0

    # Check if ear indicates a blink (threshold determined empirically)
    return ear < 0.2


# Function to analyze texture using Local Binary Patterns (LBP)
def analyze_texture(face_img):
    if face_img.size == 0:
        return False

    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # Calculate LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Calculate the histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Calculate the entropy of the histogram (more uniform for screens/prints)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))

    # Screen displays tend to have more uniform texture (higher entropy)
    return entropy < 4.5  # Threshold determined empirically


# Function to track facial movements
class MotionDetector:
    def __init__(self, history_size=10):
        self.landmark_history = []
        self.history_size = history_size

    def add_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            return

        # Store only specific landmarks for efficiency (e.g., nose tip and chin)
        key_points = [landmarks[30], landmarks[8]]  # Nose tip and chin
        self.landmark_history.append(key_points)

        # Keep only the recent history
        if len(self.landmark_history) > self.history_size:
            self.landmark_history.pop(0)

    def detect_natural_movement(self):
        if len(self.landmark_history) < self.history_size:
            return False

        # Calculate the variance of movement
        movements = []
        for i in range(1, len(self.landmark_history)):
            # Calculate movement between consecutive frames for each key point
            for p_idx in range(len(self.landmark_history[0])):
                prev_point = self.landmark_history[i - 1][p_idx]
                curr_point = self.landmark_history[i][p_idx]
                movement = distance.euclidean(prev_point, curr_point)
                movements.append(movement)

        if not movements:
            return False

        # Calculate statistics on the movements
        avg_movement = np.mean(movements)
        std_movement = np.std(movements)

        # Natural movement has some variation but isn't too large
        # These thresholds would need empirical tuning
        return 0.5 < avg_movement < 5.0 and 0.2 < std_movement < 3.0


# Main liveness detection class
class LivenessDetector:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.blink_detected = False
        self.last_blink_time = 0
        self.frame_count = 0

    def check_liveness(self, frame):
        self.frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        # No face detected
        if len(faces) == 0:
            return {
                "is_live": False,
                "confidence": 0.0,
                "face_detected": False,
                "message": "No face detected"
            }

        # Only check the first face for simplicity
        face = faces[0]

        # Get landmarks
        landmarks = get_landmarks(gray, face)

        # Extract face region
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = frame[y:y + h, x:x + w]

        # Check for eye blinks
        if detect_blink(landmarks):
            self.blink_detected = True

        # Add landmarks to motion detector
        self.motion_detector.add_landmarks(landmarks)

        # Analyze texture
        texture_real = analyze_texture(face_region)

        # Check for natural movement
        natural_movement = self.motion_detector.detect_natural_movement()

        # Combine results
        # Give more frames before requiring all checks to pass
        confidence = 0.0

        if self.frame_count < 30:  # Initial grace period
            confidence = 0.5
            message = "Analyzing..."
            is_live = None
        else:
            # Calculate confidence score based on multiple factors
            if self.blink_detected:
                confidence += 0.4

            if texture_real:
                confidence += 0.3

            if natural_movement:
                confidence += 0.3

            is_live = confidence >= 0.7  # Threshold for liveness

            if is_live:
                message = "Real face detected"
            else:
                message = "Spoof detected"

            # List what's missing
            if not self.blink_detected:
                message += " (No blink detected)"
            if not texture_real:
                message += " (Abnormal texture)"
            if not natural_movement:
                message += " (Unnatural movement)"

        return {
            "is_live": is_live,
            "confidence": confidence,
            "face_detected": True,
            "blink_detected": self.blink_detected,
            "texture_real": texture_real,
            "natural_movement": natural_movement,
            "message": message
        }

    def reset(self):
        self.motion_detector = MotionDetector()
        self.blink_detected = False
        self.frame_count = 0