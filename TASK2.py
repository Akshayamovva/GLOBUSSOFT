from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

# Initialize FastAPI app
app = FastAPI(title="Face Verification API")

# Load HaarCascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
# ORB feature detector and BFMatcher 
orb_detector = cv2.ORB_create(nfeatures=2000)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING) 

# Read uploaded file and convert to OpenCV image.

def load_image(uploaded_file: UploadFile):
    img_bytes = uploaded_file.file.read()
    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image
    
# Detect face and return cropped face + bounding box. Returns None if no face.

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        bbox = [int(x), int(y), int(x+w), int(y+h)]
        face_resized = cv2.resize(cropped_face, (224, 224))
        return face_resized, bbox
    else:
        return None, None
        
#Detect face and return cropped face + bounding box
def extract_features(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb_detector.detectAndCompute(gray_face, None)
    if keypoints is None or len(keypoints) == 0:
        descriptors = None
    return keypoints, descriptors
    
#Compute similarity score using KNN + ratio test
def match_features(descriptors1, descriptors2):
    """Compute similarity score using KNN + ratio test."""
    if descriptors1 is None or descriptors2 is None:
        return 0
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75    * n.distance]
    similarity_score = len(good_matches)
    return similarity_score

@app.post("/verify")
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # Load images
    img1 = load_image(image1)
    img2 = load_image(image2)

    # Detect faces
    face1, bbox1 = detect_face(img1)
    face2, bbox2 = detect_face(img2)

    if face1 is None or face2 is None:
        return JSONResponse({"error": "Face not detected in one or both images"})

    # Extract ORB features
    _, descriptors1 = extract_features(face1)
    _, descriptors2 = extract_features(face2)

    # Compute similarity
    similarity_score = match_features(descriptors1, descriptors2)

    # Decide if same person (tune threshold as needed)
    verification_result = "same person" if similarity_score > 30 else "different person"

    return {
        "verification_result": verification_result,
        "similarity_score": similarity_score,
        "face1_bbox": bbox1,
        "face2_bbox": bbox2
    }
