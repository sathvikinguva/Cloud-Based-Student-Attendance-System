import datetime
import os
import uuid
import logging
import json
import hashlib
import math
import numpy as np
import cv2
import random
import string
from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient
import azure.ai.vision.face.models as face_models

# Flask app setup with session support
app = Flask(__name__)
app.secret_key = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
CORS(app)  # Enable CORS for frontend-backend communication

# Create separate folders for registered and captured faces
REGISTERED_FOLDER = os.path.join(os.getcwd(), "registered_faces")
CAPTURED_FOLDER = os.path.join(os.getcwd(), "captured_faces")
MODEL_FOLDER = os.path.join(os.getcwd(), "models")
os.makedirs(REGISTERED_FOLDER, exist_ok=True)
os.makedirs(CAPTURED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Database file paths
FACE_DB_PATH = os.path.join(os.getcwd(), "face_db.json")
STUDENT_DB_PATH = os.path.join(os.getcwd(), "student_db.json")
ATTENDANCE_LOG_PATH = os.path.join(os.getcwd(), "attendance_log.json")
CHALLENGE_DB_PATH = os.path.join(os.getcwd(), "challenge_db.json")
DEVICE_DB_PATH = os.path.join(os.getcwd(), "device_db.json")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Azure Face API configuration
FACE_API_KEY = os.getenv("FACE_APIKEY", "4uEgB3aZTHDotOTRFAu8En02WA776eDKDa56KknOlsHiI1IdVMGaJQQJ99BCACGhslBXJ3w3AAAKACOGA0XX")
FACE_API_ENDPOINT = os.getenv("FACE_ENDPOINT", "https://inguva.cognitiveservices.azure.com/")

# Validate Azure Face API credentials
if not FACE_API_KEY or not FACE_API_ENDPOINT:
    logger.error("Azure Face API credentials are missing. Check environment variables.")
    raise ValueError("Azure Face API credentials are missing.")

# Azure Face API client
face_client = FaceClient(
    endpoint=FACE_API_ENDPOINT,
    credential=AzureKeyCredential(FACE_API_KEY)
)

# System configuration
SYSTEM_CONFIG = {
    "liveness_detection_enabled": True,
    "challenge_verification_enabled": True,
    "location_verification_enabled": True,
    "device_verification_enabled": True,
    "multi_factor_attendance": True,
    "min_face_quality_score": 0.5,
    "high_confidence_threshold": 0.90,
    "medium_confidence_threshold": 0.80,
    "low_confidence_threshold": 0.65,
    "max_allowed_failed_attempts": 3,
    "attendance_window_minutes": 15,  
    "max_time_between_challenges": 60,  
    "location_tolerance_meters": 50,  
    "class_locations": {
        # Format: "class_id": {"lat": latitude, "lng": longitude, "radius": radius_in_meters}
    }
}

# Initialize database files if they don't exist
def initialize_db_files():
    db_files = {
        FACE_DB_PATH: {},
        STUDENT_DB_PATH: {},
        ATTENDANCE_LOG_PATH: {},
        CHALLENGE_DB_PATH: {},
        DEVICE_DB_PATH: {}
    }
    
    for path, default_data in db_files.items():
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(default_data, f)

initialize_db_files()

PROTOTXT_PATH = os.path.join(MODEL_FOLDER, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_FOLDER, "res10_300x300_ssd_iter_140000.caffemodel")

def download_dnn_model():
    """Download DNN face detection model files if not present"""
    try:
        import urllib.request
        
        if not os.path.exists(PROTOTXT_PATH):
            logger.info("Downloading face detection prototxt file...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                PROTOTXT_PATH
            )
            
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading face detection model file...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                MODEL_PATH
            )
            
        logger.info("Face detection model files ready")
        return True
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        return False

# Try to download DNN model files
download_dnn_model()

def get_image_hash(image_data):
    """Generate a hash for an image to use for simple comparison"""
    return hashlib.md5(image_data).hexdigest()

def assess_face_quality(image_path):
    """
    Assess the quality of a face image for recognition reliability
    Returns a quality score between 0.0 and 1.0
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Check image resolution
        height, width = gray.shape
        resolution_score = min(1.0, (height * width) / (640 * 480))
        
        # 2. Check image brightness and contrast
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        std_dev = np.std(gray)
        contrast_score = min(1.0, std_dev / 64)
        
        # 3. Check for blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500)
        
        # 4. Face detection confidence
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_detection_score = 0.0
        if len(faces) > 0:
            face_detection_score = min(1.0, len(faces))
            
        # Calculate weighted quality score
        quality_score = (0.2 * resolution_score + 
                        0.2 * brightness_score + 
                        0.3 * contrast_score + 
                        0.3 * blur_score)
        
        logger.info(f"Face quality assessment for {image_path}: {quality_score:.2f}")
        return quality_score
        
    except Exception as e:
        logger.error(f"Error assessing face quality: {str(e)}")
        return 0.0

def detect_liveness(image_path):
    """
    Detect if the face is from a live person and not a photo
    Returns (is_live, confidence_score)
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return False, 0.0

        # Basic liveness checks
        
        # 1. Convert to different color spaces to look for telling signs of printouts/screens
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # 2. Check for moire patterns that appear when photographing screens
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply high-pass filter to detect screen patterns
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        highpass = cv2.filter2D(gray, -1, kernel)
        
        # Calculate standard deviation in high-frequency components
        highfreq_std = np.std(highpass)
        
        # 3. Check color distribution in different channels
        b, g, r = cv2.split(image)
        h, s, v = cv2.split(hsv)
        y, cr, cb = cv2.split(ycrcb)
        
        # Real faces have specific distributions in these colorspaces
        cr_cb_ratio = np.mean(cr) / (np.mean(cb) + 1e-5)
        s_v_ratio = np.mean(s) / (np.mean(v) + 1e-5)
        bg_ratio = np.mean(b) / (np.mean(g) + 1e-5)
        
        # 4. Check texture variance (real faces have more texture)
        texture_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        texture = cv2.filter2D(gray, -1, texture_kernel)
        texture_variance = np.var(texture)
        
        # 5. Calculate gradients to detect edges (printed faces often have sharper edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        edges = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
        edge_intensity = np.mean(edges)
        
        # Combine all factors into a liveness score
        # These thresholds are approximate and should be tuned based on testing
        factors = {
            "highfreq_std": min(1.0, highfreq_std / 30),  # Lower for printed images
            "cr_cb_ratio": min(1.0, abs(cr_cb_ratio - 1.6) / 0.6),  # Typical live face is around 1.6
            "texture_variance": min(1.0, texture_variance / 1000),  # Higher for real faces
            "edge_intensity": min(1.0, (70 - edge_intensity) / 70)  # Lower for real faces (smoother)
        }
        
        liveness_score = (
            0.3 * factors["highfreq_std"] +
            0.3 * factors["cr_cb_ratio"] +
            0.2 * factors["texture_variance"] +
            0.2 * factors["edge_intensity"]
        )
        
        # Output all factors for debugging
        logger.debug(f"Liveness factors: {factors}, Score: {liveness_score:.2f}")
        
        # Consider score > 0.7 as live
        is_live = liveness_score > 0.7
        
        return is_live, liveness_score
        
    except Exception as e:
        logger.error(f"Error in liveness detection: {str(e)}")
        return False, 0.0

def detect_and_extract_facial_landmarks(image_path):
    """Extract facial landmarks to use for liveness detection and face matching"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Use dlib's face detector and predictor if available
        try:
            import dlib # type: ignore
            detector = dlib.get_frontal_face_detector()
            # Check if predictor file exists, if not download
            predictor_path = os.path.join(MODEL_FOLDER, "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(predictor_path):
                logger.warning("Facial landmark predictor not found, falling back to basic detection")
                raise ImportError("Landmark predictor not available")
                
            predictor = dlib.shape_predictor(predictor_path)
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector(gray)
            
            if len(faces) == 0:
                logger.warning("No face detected for landmark extraction")
                return None
                
            # Get the first face
            face = faces[0]
            
            # Predict landmarks
            shape = predictor(gray, face)
            
            # Extract landmark coordinates
            landmarks = []
            for i in range(68):  # There are 68 landmarks
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))
                
            return landmarks
            
        except (ImportError, Exception) as e:
            logger.warning(f"Could not use dlib for landmarks: {str(e)}")
            
            # Fallback - use OpenCV's facial landmark detection if available
            try:
                # Try to use OpenCV's face module
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) == 0:
                    logger.warning("No face detected for fallback landmark extraction")
                    return None
                    
                # Take the first face
                (x, y, w, h) = faces[0]
                
                # Create basic landmarks (just face corners and center)
                # This is much less accurate than dlib but provides some data
                landmarks = [
                    (x, y),                 # Top-left
                    (x + w, y),             # Top-right
                    (x, y + h),             # Bottom-left
                    (x + w, y + h),         # Bottom-right
                    (x + w//2, y + h//2),   # Center
                    (x + w//4, y + h//3),   # Left eye (approx)
                    (x + 3*w//4, y + h//3), # Right eye (approx)
                    (x + w//2, y + 2*h//3)  # Mouth (approx)
                ]
                return landmarks
                
            except Exception as e:
                logger.error(f"Error in fallback landmark detection: {str(e)}")
                return None
                
    except Exception as e:
        logger.error(f"Error in facial landmark detection: {str(e)}")
        return None

def extract_face_features(image_path):
    """
    Extract facial features from an image using OpenCV
    Returns a feature vector for face comparison
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Convert to grayscale for better feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try DNN face detection first (more accurate)
        face_detected = False
        face_roi = None
        
        if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
            try:
                # Load DNN face detector
                net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
                
                # Preprocess image for face detection
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                            (300, 300), (104.0, 177.0, 123.0))
                
                # Detect faces
                net.setInput(blob)
                detections = net.forward()
                
                # Find detection with highest confidence
                max_confidence = 0
                max_confidence_idx = -1
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > max_confidence:
                        max_confidence = confidence
                        max_confidence_idx = i
                
                # If a face with good confidence is detected
                if max_confidence_idx >= 0 and max_confidence > 0.5:
                    # Extract face bounding box
                    box = detections[0, 0, max_confidence_idx, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure box is within image boundaries
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # Extract face ROI
                    face_roi = gray[startY:endY, startX:endX]
                    face_detected = True
                    logger.info(f"Face detected with DNN model, confidence: {max_confidence:.2f}")
            except Exception as e:
                logger.warning(f"DNN face detection failed: {str(e)}")
        
        # Fall back to Haar cascade if DNN detection failed
        if not face_detected:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logger.info(f"No face detected in image: {image_path}")
                # Return image histogram as fallback
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                return hist.tolist()
                
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            logger.info("Face detected with Haar cascade")
        
        # If we reach here, we should have a valid face_roi
        if face_roi is None or face_roi.size == 0:
            logger.warning("Invalid face region extracted")
            # Return image histogram as fallback
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist.tolist()
            
        # Resize to a standard size for comparison
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Create feature vector from multiple techniques
        
        # 1. Histogram features (basic but effective)
        hist = cv2.calcHist([face_roi], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 2. Local Binary Patterns (LBP)
        lbp_features = []
        radius = 1
        neighbors = 8
        
        # Simple LBP implementation
        for i in range(radius, face_roi.shape[0]-radius):
            for j in range(radius, face_roi.shape[1]-radius):
                center = face_roi[i, j]
                pattern = 0
                for k in range(neighbors):
                    # Calculate neighbor pixel coordinates
                    x = i + int(radius * math.cos(2 * math.pi * k / neighbors))
                    y = j - int(radius * math.sin(2 * math.pi * k / neighbors))
                    
                    # Add to pattern
                    if face_roi[x, y] >= center:
                        pattern |= (1 << k)
                
                lbp_features.append(pattern)
        
        # Create LBP histogram
        lbp_hist = np.bincount(lbp_features, minlength=256)
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= np.sum(lbp_hist) + 1e-6  # Normalize
        
        # 3. Image moments for shape features
        moments = cv2.moments(face_roi)
        moment_features = [moments['m00'], moments['m10'], moments['m01'], 
                          moments['m20'], moments['m11'], moments['m02']]
                          
        # 4. Horizontal and vertical projections
        h_proj = np.sum(face_roi, axis=1) / face_roi.shape[1]  # Horizontal projection
        v_proj = np.sum(face_roi, axis=0) / face_roi.shape[0]  # Vertical projection
        
        # Get facial landmarks if possible
        landmarks = detect_and_extract_facial_landmarks(image_path)
        landmark_features = []
        
        if landmarks:
            # Calculate relative positions of landmarks
            center_x = face_roi.shape[1] / 2
            center_y = face_roi.shape[0] / 2
            
            # Normalize landmarks relative to face center and size
            for (x, y) in landmarks[:20]:  # Use first 20 landmarks at most
                norm_x = (x - center_x) / (face_roi.shape[1] / 2)
                norm_y = (y - center_y) / (face_roi.shape[0] / 2)
                landmark_features.extend([norm_x, norm_y])
                
            # Pad to fixed length if needed
            while len(landmark_features) < 40:  # 20 landmarks * 2 coordinates
                landmark_features.append(0.0)
                
            # Trim if too many
            landmark_features = landmark_features[:40]
        else:
            # No landmarks detected, pad with zeros
            landmark_features = [0.0] * 40
        
        # Combine all features (use most important ones)
        # Using fewer features can actually make comparison more robust
        features = np.concatenate((
            hist[:32],  # Use first 32 histogram bins
            lbp_hist[::4][:32],  # Sample LBP histogram
            np.array(moment_features),
            h_proj[::5][:10],  # Sample projections to reduce dimensions
            v_proj[::5][:10],
            np.array(landmark_features)[:40]  # Add landmark features if available
        ))
        
        return features.tolist()  # Convert to list for JSON serialization
        
    except Exception as e:
        logger.error(f"Error extracting face features: {e}", exc_info=True)
        return None

def calculate_face_similarity(vec1, vec2):
    """Calculate similarity between two face vectors using multiple metrics"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
        
    # 1. Cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 * magnitude2 == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    # 2. Euclidean distance (converted to similarity)
    euclidean_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    euclidean_sim = 1.0 / (1.0 + euclidean_dist)
    
    # 3. Manhattan distance (converted to similarity)
    manhattan_dist = sum(abs(a - b) for a, b in zip(vec1, vec2))
    manhattan_sim = 1.0 / (1.0 + manhattan_dist)
    
    # Weighted average of similarity metrics
    # Cosine similarity is given highest weight as it works well for face features
    combined_sim = (0.6 * cosine_sim) + (0.25 * euclidean_sim) + (0.15 * manhattan_sim)
    
    logger.debug(f"Similarity metrics - cosine: {cosine_sim:.4f}, euclidean: {euclidean_sim:.4f}, manhattan: {manhattan_sim:.4f}, combined: {combined_sim:.4f}")
    
    return combined_sim

def get_registered_face_by_student_id(student_id):
    """Get the registered face details for a specific student"""
    try:
        # Load student database
        with open(STUDENT_DB_PATH, 'r') as f:
            student_db = json.load(f)
            
        if student_id not in student_db:
            logger.error(f"Student ID {student_id} not found in database")
            return None
            
        # Get the face ID for this student
        face_id = student_db[student_id].get("face_id")
        
        if not face_id:
            logger.error(f"No face registered for student ID {student_id}")
            return None
            
        # Load face database
        with open(FACE_DB_PATH, 'r') as f:
            face_db = json.load(f)
            
        if face_id not in face_db:
            logger.error(f"Face ID {face_id} not found in face database")
            return None
            
        return face_db[face_id]
            
    except Exception as e:
        logger.error(f"Error getting registered face: {str(e)}")
        return None

def generate_challenge():
    """Generate a random challenge for liveness verification"""
    challenges = [
        {"type": "smile", "instruction": "Please smile for the camera"},
        {"type": "blink", "instruction": "Please blink twice"},
        {"type": "turn_left", "instruction": "Please turn your head slightly to the left"},
        {"type": "turn_right", "instruction": "Please turn your head slightly to the right"},
        {"type": "nod", "instruction": "Please nod your head"}
    ]
    
    # Select a random challenge
    challenge = random.choice(challenges)
    
    # Add a unique ID and timestamp
    challenge["id"] = str(uuid.uuid4())
    challenge["timestamp"] = datetime.datetime.now().timestamp()
    
    return challenge

def verify_device(device_id, student_id):
    """Verify if the device is registered to the student"""
    try:
        if not device_id or not student_id:
            return False, "Missing device ID or student ID"
            
        # Load device database
        with open(DEVICE_DB_PATH, 'r') as f:
            device_db = json.load(f)
            
        # Check if device exists and is associated with the student
        if device_id not in device_db:
            return False, f"Device {device_id} not registered"
            
        device_info = device_db[device_id]
        
        # Check if this device is registered to this student
        if device_info.get("student_id") != student_id:
            return False, f"Device {device_id} not registered to student {student_id}"
            
        return True, "Device verified"
        
    except Exception as e:
        logger.error(f"Error verifying device: {str(e)}")
        return False, f"Device verification error: {str(e)}"

def verify_location(student_lat, student_lng, class_id):
    """Verify if the student's location matches the class location"""
    try:
        if not SYSTEM_CONFIG["location_verification_enabled"]:
            return True, "Location verification disabled"
            
        if class_id not in SYSTEM_CONFIG["class_locations"]:
            return True, f"No location configured for class {class_id}"
            
        class_location = SYSTEM_CONFIG["class_locations"][class_id]
        
        # Calculate distance between student and class location
        import geopy.distance # type: ignore
        student_coords = (student_lat, student_lng)
        class_coords = (class_location["lat"], class_location["lng"])
        
        distance = geopy.distance.distance(student_coords, class_coords).meters
        
        # If distance is within tolerance, return success
        if distance <= SYSTEM_CONFIG["location_tolerance_meters"]:
            return True, f"Location verified (distance: {distance:.2f}m)"
        else:
            return False, f"Location too far from class ({distance:.2f}m)"
            
    except Exception as e:
        logger.error(f"Error verifying location: {str(e)}")
        return False, f"Location verification error: {str(e)}"

def verify_challenge_response(challenge_id, response_image_path):
    """
    Verify if the response image satisfies the challenge
    Returns (success, confidence_score, message)
    """
    try:
        # Load challenge database
        with open(CHALLENGE_DB_PATH, 'r') as f:
            challenge_db = json.load(f)
            
        if challenge_id not in challenge_db:
            return False, 0.0, f"Challenge ID {challenge_id} not found"
            
        challenge = challenge_db[challenge_id]
        
        # Check if the challenge has expired
        current_time = datetime.datetime.now().timestamp()
        time_diff = current_time - challenge["timestamp"]
        
        if time_diff > SYSTEM_CONFIG["max_time_between_challenges"]:
            return False, 0.0, f"Challenge expired ({time_diff:.1f}s > {SYSTEM_CONFIG['max_time_between_challenges']}s)"
        
        # Perform liveness detection
        is_live, liveness_score = detect_liveness(response_image_path)
        
        if not is_live:
            return False, liveness_score, "Liveness check failed, possible spoof attack"
        
        # Check challenge-specific requirements
        challenge_type = challenge["type"]
        success = False
        confidence = 0.7  # Base confidence
        
        if challenge_type == "smile":
            # Here you would analyze for a smile
            # For now, we'll assume it passed if the image is live
            success = True
            message = "Smile detected"
            
        elif challenge_type == "blink":
            # Here you would analyze for blinks
            success = True
            message = "Blink detected"
            
        elif challenge_type == "turn_left" or challenge_type == "turn_right":
            # Here you would analyze head pose
            success = True
            message = "Head turn detected"
            
        elif challenge_type == "nod":
            # Here you would analyze for head nodding
            success = True
            message = "Nod detected"
            
        else:
            success = True  # Default to success for unimplemented challenge types
            message = "Challenge response accepted"
        
        return success, confidence, message
        
    except Exception as e:
        logger.error(f"Error verifying challenge response: {str(e)}")
        return False, 0.0, f"Challenge verification error: {str(e)}"

def log_attendance(student_id, class_id, status, confidence, method, location=None, device_id=None):
    """Log an attendance record with detailed information"""
    try:
        # Load attendance log
        with open(ATTENDANCE_LOG_PATH, 'r') as f:
            attendance_log = json.load(f)
        
        # Create a log entry
        entry_id = str(uuid.uuid4())
        timestamp = str(datetime.datetime.now())
        
        entry = {
            "id": entry_id,
            "student_id": student_id,
            "class_id": class_id,
            "status": status,  # "present", "absent", "late", etc.
            "confidence": confidence,
            "verification_method": method,
            "timestamp": timestamp,
            "location": location,
            "device_id": device_id
        }
        
        # Add to attendance log
        if "entries" not in attendance_log:
            attendance_log["entries"] = []
            
        attendance_log["entries"].append(entry)
        
        # Save back to file
        with open(ATTENDANCE_LOG_PATH, 'w') as f:
            json.dump(attendance_log, f)
            
        return entry_id
        
    except Exception as e:
        logger.error(f"Error logging attendance: {str(e)}")
        return None

@app.route("/api/register-student", methods=["POST"])
def register_student():
    """Register a new student with their face"""
    file_path = None
    try:
        # Get student ID and optional device ID
        student_id = request.form.get('studentId')
        device_id = request.form.get('deviceId')
        
        if not student_id:
            return jsonify({"error": "Student ID is required"}), 400
        
        # Check if an image file is provided
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file to the registered faces folder
        filename = f"student_{student_id}_{uuid.uuid4()}.jpg"
        file_path = os.path.join(REGISTERED_FOLDER, filename)
        file.save(file_path)

        logger.info(f"Registered face saved at: {file_path}")

        # Check face image quality
        quality_score = assess_face_quality(file_path)
        if quality_score < SYSTEM_CONFIG["min_face_quality_score"]:
            return jsonify({
                "error": "Poor quality face image",
                "details": "The uploaded image has low quality (poor lighting, blurry, or low resolution)",
                "qualityScore": quality_score
            }), 400

        # Perform liveness detection
        is_live, liveness_score = detect_liveness(file_path)
        if SYSTEM_CONFIG["liveness_detection_enabled"] and not is_live:
            return jsonify({
                "error": "Liveness check failed",
                "details": "The provided image does not appear to be of a live person.",
                "livenessScore": liveness_score
            }), 400

        # Extract face features
        features = extract_face_features(file_path)
        if not features:
            return jsonify({"error": "Could not extract face features"}), 400

        # Generate a face ID
        face_id = f"face_{uuid.uuid4()}"
        
        # Read file for hash
        with open(file_path, 'rb') as f:
            image_data = f.read()
            image_hash = get_image_hash(image_data)
        
        # Store the face ID and features in face database
        try:
            with open(FACE_DB_PATH, 'r') as f:
                face_db = json.load(f)
        except:
            face_db = {}
        
        # Store face data
        face_db[face_id] = {
            "hash": image_hash,
            "features": features,
            "student_id": student_id,
            "is_registered": True,
            "file_path": file_path,
            "quality_score": quality_score,
            "liveness_score": liveness_score,
            "timestamp": str(datetime.datetime.now())
        }
        
        with open(FACE_DB_PATH, 'w') as f:
            json.dump(face_db, f)
            
        # Update student database with face ID
        try:
            with open(STUDENT_DB_PATH, 'r') as f:
                student_db = json.load(f)
        except:
            student_db = {}
            
        # Store or update student data
        student_db[student_id] = {
            "face_id": face_id,
            "registered_at": str(datetime.datetime.now()),
            "last_verified": None,
            "face_quality": quality_score,
            "registered_devices": []
        }
        
        with open(STUDENT_DB_PATH, 'w') as f:
            json.dump(student_db, f)
            
        # If device ID provided, register it
        if device_id:
            try:
                with open(DEVICE_DB_PATH, 'r') as f:
                    device_db = json.load(f)
            except:
                device_db = {}
                
            # Register device to this student
            device_db[device_id] = {
                "student_id": student_id,
                "registered_at": str(datetime.datetime.now()),
                "last_used": str(datetime.datetime.now())
            }
            
            with open(DEVICE_DB_PATH, 'w') as f:
                json.dump(device_db, f)
                
            # Add device to student's registered devices
            student_db[student_id]["registered_devices"].append(device_id)
            
            with open(STUDENT_DB_PATH, 'w') as f:
                json.dump(student_db, f)

        logger.info(f"Registered student {student_id} with face ID: {face_id}, quality: {quality_score:.2f}")

        return jsonify({
            "faceId": face_id, 
            "message": "Student registered successfully",
            "qualityScore": quality_score,
            "livenessScore": liveness_score
        })

    except Exception as e:
        logger.error(f"Error in student registration: {str(e)}", exc_info=True)
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route("/api/request-challenge", methods=["POST"])
def request_challenge():
    """Request a challenge for attendance verification"""
    try:
        # Get student ID and class ID
        data = request.get_json() or {}
        student_id = data.get("studentId") or request.form.get("studentId")
        class_id = data.get("classId") or request.form.get("classId")
        
        if not student_id or not class_id:
            return jsonify({"error": "Student ID and Class ID are required"}), 400
        
        # Generate a challenge
        challenge = generate_challenge()
        
        # Store the challenge with student and class info
        challenge["student_id"] = student_id
        challenge["class_id"] = class_id
        
        try:
            with open(CHALLENGE_DB_PATH, 'r') as f:
                challenge_db = json.load(f)
        except:
            challenge_db = {}
            
        # Store challenge
        challenge_db[challenge["id"]] = challenge
        
        with open(CHALLENGE_DB_PATH, 'w') as f:
            json.dump(challenge_db, f)
        
        # Return challenge to client
        return jsonify({
            "challengeId": challenge["id"],
            "type": challenge["type"],
            "instruction": challenge["instruction"]
        })
        
    except Exception as e:
        logger.error(f"Error requesting challenge: {str(e)}")
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route("/api/detect-face", methods=["POST"])
def detect_face():
    """Detect a face in a captured image for attendance"""
    file_path = None
    try:
        # Get student ID if provided, as well as challenge, class, device, location info
        student_id = request.form.get('studentId')
        challenge_id = request.form.get('challengeId')
        class_id = request.form.get('classId')
        device_id = request.form.get('deviceId')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        # Check if an image file is provided
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file to the captured faces folder
        filename = f"captured_{uuid.uuid4()}.jpg"
        file_path = os.path.join(CAPTURED_FOLDER, filename)
        file.save(file_path)

        logger.info(f"Captured face saved at: {file_path}")

        # Check face quality
        quality_score = assess_face_quality(file_path)
        if quality_score < SYSTEM_CONFIG["min_face_quality_score"]:
            return jsonify({
                "error": "Poor quality face image",
                "details": "The captured image has low quality (poor lighting, blurry, or low resolution)",
                "qualityScore": quality_score
            }), 400
            
        # Perform liveness detection
        is_live, liveness_score = detect_liveness(file_path)
        
        # If this is a challenge response, verify the challenge
        challenge_verified = False
        challenge_message = "No challenge provided"
        
        if challenge_id and SYSTEM_CONFIG["challenge_verification_enabled"]:
            challenge_verified, challenge_confidence, challenge_message = verify_challenge_response(challenge_id, file_path)
            if not challenge_verified:
                return jsonify({
                    "error": "Challenge verification failed",
                    "details": challenge_message,
                    "challengeId": challenge_id
                }), 400
        
        # Extract face features
        features = extract_face_features(file_path)
        if not features:
            return jsonify({"error": "Could not extract face features"}), 400

        # Read file for hash
        with open(file_path, 'rb') as f:
            image_data = f.read()
            image_hash = get_image_hash(image_data)

        # Generate a face ID
        face_id = f"face_{uuid.uuid4()}"
        
        # Store the face ID and features in our database
        try:
            with open(FACE_DB_PATH, 'r') as f:
                face_db = json.load(f)
        except:
            face_db = {}
        
        # Store face data
        face_db[face_id] = {
            "hash": image_hash,
            "features": features,
            "student_id": student_id,  # Store student ID if provided
            "is_registered": False,  # This is a captured face, not registered
            "file_path": file_path,
            "quality_score": quality_score,
            "liveness_score": liveness_score,
            "challenge_verified": challenge_verified,
            "timestamp": str(datetime.datetime.now())
        }
        
        with open(FACE_DB_PATH, 'w') as f:
            json.dump(face_db, f)

        logger.info(f"Detected face with ID: {face_id}, quality: {quality_score:.2f}, liveness: {liveness_score:.2f}")
        
        # If student ID is provided, immediately verify against registered face
        if student_id:
            # First, verify the device if device verification is enabled
            device_verified = True
            device_message = "Device verification disabled"
            
            if device_id and SYSTEM_CONFIG["device_verification_enabled"]:
                device_verified, device_message = verify_device(device_id, student_id)
                if not device_verified:
                    return jsonify({
                        "error": "Device verification failed",
                        "details": device_message,
                        "faceId": face_id,
                        "deviceId": device_id
                    }), 400
            
            # Next, verify location if location verification is enabled
            location_verified = True
            location_message = "Location verification disabled"
            
            if latitude and longitude and class_id and SYSTEM_CONFIG["location_verification_enabled"]:
                location_verified, location_message = verify_location(float(latitude), float(longitude), class_id)
                if not location_verified:
                    return jsonify({
                        "error": "Location verification failed",
                        "details": location_message,
                        "faceId": face_id
                    }), 400
            
            # Finally, verify against the registered face
            registered_face = get_registered_face_by_student_id(student_id)
            
            if registered_face and "features" in registered_face:
                # Compare the features using enhanced similarity function
                similarity = calculate_face_similarity(registered_face["features"], features)
                logger.info(f"Immediate verification result: similarity={similarity}")
                
                # Use multiple thresholds for different confidence levels
                high_threshold = SYSTEM_CONFIG["high_confidence_threshold"]
                medium_threshold = SYSTEM_CONFIG["medium_confidence_threshold"]
                low_threshold = SYSTEM_CONFIG["low_confidence_threshold"]
                
                # Determine confidence level
                if similarity >= high_threshold:
                    is_identical = True
                    confidence_level = "high"
                elif similarity >= medium_threshold:
                    is_identical = True
                    confidence_level = "medium"
                elif similarity >= low_threshold:
                    is_identical = True
                    confidence_level = "low"
                else:
                    is_identical = False
                    confidence_level = "insufficient"
                
                # If all verifications pass, log attendance
                if is_identical:
                    attendance_status = "present"
                    
                    # Log attendance with all verification details
                    location_data = None
                    if latitude and longitude:
                        location_data = {"lat": float(latitude), "lng": float(longitude)}
                    
                    verification_methods = []
                    if SYSTEM_CONFIG["liveness_detection_enabled"] and is_live:
                        verification_methods.append("liveness")
                    if challenge_verified:
                        verification_methods.append("challenge")
                    if device_verified:
                        verification_methods.append("device")
                    if location_verified:
                        verification_methods.append("location")
                    verification_methods.append("face_recognition")
                    
                    log_id = log_attendance(
                        student_id=student_id,
                        class_id=class_id,
                        status=attendance_status,
                        confidence=similarity,
                        method=",".join(verification_methods),
                        location=location_data,
                        device_id=device_id
                    )
                    
                    # Include attendance log ID in verification result
                    return jsonify([{
                        "faceId": face_id,
                        "qualityScore": quality_score,
                        "livenessScore": liveness_score,
                        "verificationResult": {
                            "isIdentical": is_identical,
                            "confidence": similarity,
                            "confidenceLevel": confidence_level,
                            "registeredFaceId": registered_face.get("face_id")
                        },
                        "attendanceStatus": attendance_status,
                        "attendanceLogId": log_id,
                        "verifications": {
                            "liveness": is_live,
                            "challenge": challenge_verified,
                            "device": device_verified,
                            "location": location_verified
                        }
                    }])
                else:
                    # Identity verification failed
                    return jsonify([{
                        "faceId": face_id,
                        "qualityScore": quality_score,
                        "livenessScore": liveness_score,
                        "verificationResult": {
                            "isIdentical": False,
                            "confidence": similarity,
                            "confidenceLevel": "insufficient",
                            "message": "Face doesn't match registered student"
                        },
                        "attendanceStatus": "absent",
                        "verifications": {
                            "liveness": is_live,
                            "challenge": challenge_verified,
                            "device": device_verified,
                            "location": location_verified
                        }
                    }])

        # Return the face ID and quality score
        return jsonify([{
            "faceId": face_id,
            "qualityScore": quality_score,
            "livenessScore": liveness_score,
            "message": "Face detected but no verification performed"
        }])

    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}", exc_info=True)
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route("/api/verify-face", methods=["POST"])
def verify_face():
    """Verify if a captured face matches the registered face for a student"""
    try:
        # Get the request data
        data = request.get_json(silent=True)
        logger.info(f"Received verification request with data: {data}")
        
        # Try to get data from either JSON or form data
        if data:
            face_id1 = data.get("faceId1")  # Registered face ID
            face_id2 = data.get("faceId2")  # Captured face ID
            student_id = data.get("studentId")
            class_id = data.get("classId")
            device_id = data.get("deviceId")
            latitude = data.get("latitude")
            longitude = data.get("longitude")
        else:
            # Try form data
            face_id1 = request.form.get("faceId1")
            face_id2 = request.form.get("faceId2")
            student_id = request.form.get("studentId")
            class_id = request.form.get("classId")
            device_id = request.form.get("deviceId")
            latitude = request.form.get("latitude")
            longitude = request.form.get("longitude")
            
        logger.info(f"Extracted data: face_id1={face_id1}, face_id2={face_id2}, student_id={student_id}, class_id={class_id}")
        
        # Load face database
        try:
            with open(FACE_DB_PATH, 'r') as f:
                face_db = json.load(f)
        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
            return jsonify({
                "isIdentical": False,
                "confidence": 0.1,
                "message": "Error loading face database"
            }), 500
            
        # If we have a student ID but no registered face ID, get it from the student database
        if student_id and not face_id1:
            try:
                with open(STUDENT_DB_PATH, 'r') as f:
                    student_db = json.load(f)
                    
                if student_id in student_db:
                    face_id1 = student_db[student_id].get("face_id")
                    logger.info(f"Found registered face ID {face_id1} for student {student_id}")
            except Exception as e:
                logger.error(f"Error loading student database: {str(e)}")
        
        # Validate parameters
        if not face_id2:
            logger.error("Missing captured face ID")
            return jsonify({
                "isIdentical": False,
                "confidence": 0.1,
                "message": "Captured face ID is required"
            }), 400

        if face_id2 not in face_db:
            logger.error(f"Captured face ID {face_id2} not found in database")
            return jsonify({
                "isIdentical": False,
                "confidence": 0.1,
                "message": "Captured face not found"
            }), 404
            
        captured_face = face_db[face_id2]
        
        liveness_score = captured_face.get("liveness_score", 0.0)
        if SYSTEM_CONFIG["liveness_detection_enabled"] and liveness_score < 0.7:
            return jsonify({
                "isIdentical": False,
                "confidence": 0.0,
                "message": f"Liveness check failed for captured face (score: {liveness_score:.2f})",
                "livenessScore": liveness_score
            }), 400
        
        device_verified = True
        device_message = "Device verification disabled"
        
        if device_id and student_id and SYSTEM_CONFIG["device_verification_enabled"]:
            device_verified, device_message = verify_device(device_id, student_id)
            if not device_verified:
                return jsonify({
                    "isIdentical": False,
                    "confidence": 0.0,
                    "message": device_message,
                    "deviceVerification": {
                        "verified": False,
                        "message": device_message
                    }
                }), 400
        
        location_verified = True
        location_message = "Location verification disabled"
        
        if latitude and longitude and class_id and SYSTEM_CONFIG["location_verification_enabled"]:
            try:
                lat_float = float(latitude)
                lng_float = float(longitude)
                location_verified, location_message = verify_location(lat_float, lng_float, class_id)
                if not location_verified:
                    return jsonify({
                        "isIdentical": False,
                        "confidence": 0.0,
                        "message": location_message,
                        "locationVerification": {
                            "verified": False,
                            "message": location_message
                        }
                    }), 400
            except Exception as e:
                logger.error(f"Error processing location: {str(e)}")
                location_verified = False
                location_message = f"Location processing error: {str(e)}"
        
        if face_id1 and face_id1 in face_db:
            registered_face = face_db[face_id1]
            logger.info(f"Comparing face {face_id1} with face {face_id2}")

        elif student_id:
            registered_face = None
            for face_id, face_data in face_db.items():
                if face_data.get("student_id") == student_id and face_data.get("is_registered") == True:
                    registered_face = face_data
                    face_id1 = face_id
                    logger.info(f"Found registered face {face_id1} for student {student_id}")
                    break
                    
            if not registered_face:
                logger.error(f"No registered face found for student {student_id}")
                return jsonify({
                    "isIdentical": False,
                    "confidence": 0.1,
                    "message": f"No registered face found for student {student_id}"
                })
        else:
            logger.error("Missing both registered face ID and student ID")
            return jsonify({
                "isIdentical": False,
                "confidence": 0.1,
                "message": "Either registered face ID or student ID is required"
            }), 400

        if "features" not in registered_face or "features" not in captured_face:
            logger.error("Face features missing")
            return jsonify({
                "isIdentical": False,
                "confidence": 0.1,
                "message": "Face features missing for comparison"
            })
    
        similarity = calculate_face_similarity(registered_face["features"], captured_face["features"])
        logger.info(f"Face similarity: {similarity}")
        
        reg_quality = registered_face.get("quality_score", 0.7)
        cap_quality = captured_face.get("quality_score", 0.7)
        
        quality_factor = min(reg_quality, 0.8)

        base_high_threshold = SYSTEM_CONFIG["high_confidence_threshold"]
        base_med_threshold = SYSTEM_CONFIG["medium_confidence_threshold"]
        base_low_threshold = SYSTEM_CONFIG["low_confidence_threshold"]
        
        quality_factor = min(reg_quality, cap_quality)
        high_confidence_threshold = base_high_threshold - (quality_factor * 0.1)
        medium_confidence_threshold = base_med_threshold - (quality_factor * 0.1)
        low_confidence_threshold = base_low_threshold - (quality_factor * 0.1)

        logger.info(f"Quality factor: {quality_factor}")
        logger.info(f"Adjusted thresholds - High: {high_confidence_threshold}, Medium: {medium_confidence_threshold}, Low: {low_confidence_threshold}")

        if similarity >= high_confidence_threshold:
            is_identical = True
            confidence_level = "high"
            logger.info("Confidence level: HIGH")
        elif similarity >= medium_confidence_threshold:
            is_identical = True
            confidence_level = "medium"
            logger.info("Confidence level: MEDIUM")
        elif similarity >= low_confidence_threshold:
            is_identical = True
            confidence_level = "low"
            logger.info("Confidence level: LOW")
        else:
            is_identical = False
            confidence_level = "insufficient"
            logger.info("Confidence level: INSUFFICIENT - Faces do not match")

        attendance_log_id = None
        if is_identical and class_id and student_id:
           attendance_status = "present"
           logger.info(f"Attendance marked as PRESENT for student {student_id}")
        else:
          attendance_status = "absent"
          logger.info(f"Attendance marked as ABSENT for student {student_id}")

        response = {
            "isIdentical": is_identical,
            "confidence": float(similarity),
            "confidenceLevel": confidence_level,
            "attendanceStatus": attendance_status,
            "qualityScores": {
                "registered": reg_quality,
                "captured": cap_quality
            },
            "thresholds": {
                "high": high_confidence_threshold,
                "medium": medium_confidence_threshold,
                "low": low_confidence_threshold
            },
            "verifications": {
                "liveness": liveness_score >= 0.7,
                "device": device_verified,
                "location": location_verified
            }
        }
        
        logger.info(f"Returning verification response: {response}")
        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Verification error: {str(e)}", exc_info=True)
        return jsonify({
            "isIdentical": False,
            "confidence": 0.0,
            "message": f"Verification error: {str(e)}"
        })

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": str(datetime.datetime.now()),
        "config": {
            "liveness_detection": SYSTEM_CONFIG["liveness_detection_enabled"],
            "challenge_verification": SYSTEM_CONFIG["challenge_verification_enabled"],
            "location_verification": SYSTEM_CONFIG["location_verification_enabled"],
            "device_verification": SYSTEM_CONFIG["device_verification_enabled"]
        }
    })

if __name__ == "__main__":
    logger.info("Starting attendance system API...")
    app.run(host="0.0.0.0", port=5000, debug=True)