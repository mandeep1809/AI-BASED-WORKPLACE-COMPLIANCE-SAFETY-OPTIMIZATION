import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from sqlalchemy import create_engine, text
from urllib.parse import quote
import pandas as pd

# Database connection
user = "user"
pw = quote("Mandeep@123")
db = "workplace"
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

import pandas as pd
import cv2
import numpy as np

# Read image using OpenCV
image_path = "E:\Mandeep\360 DigiTMG\PROJECTS\OWCS\DATA SETS\Researched_Dataset\ALL FRAMES"  # Replace with your image path
image = cv2.imread(image_path)

# Convert image to a 1D array (flattened) and store in DataFrame
image_flattened = image.flatten()
df_image = pd.DataFrame([image_flattened])

print(df_image.head())  # Display DataFrame



# Display the first few rows of the DataFrame
data.head()

# Write the data from the DataFrame to the MySQL database table named 'Saftey'
data.to_sql('Saftey', con=engine, if_exists='replace', chunksize=1000, index=False)

# Read data from the 'groceries' table in the database into a pandas DataFrame
sql = 'select * from Saftey;'
Saftey = pd.read_sql_query(sql, con=engine)














st.title("YOLO Pose & Object Detection")

# Load YOLO Models
pose_model = YOLO("E:/Mandeep/360 DigiTMG/PROJECTS/OWCS/CODES/Deployment/yolo11n-pose.pt")  # Pose estimation model
object_model = YOLO("E:/Mandeep/360 DigiTMG/PROJECTS/OWCS/CODES/Deployment/best.pt")  # Object detection model

# File uploader
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Display Table from Database
if st.button("Show Database Table"):
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM your_table_name"))  # Replace with actual table name
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Function to calculate angles between three keypoints
def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Pose classification function
def classify_pose(keypoints):
    if keypoints is None or len(keypoints) < 9:
        return "Unknown"
    
    nose, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = keypoints[:9]
    if None in (left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle):
        return "Unknown"
    
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    arm_angle = calculate_angle(left_shoulder, nose, right_shoulder)

    if None in (left_knee_angle, right_knee_angle, torso_angle, arm_angle):
        return "Unknown"
    
    stride_length = abs(left_ankle[0] - right_ankle[0])
    knee_difference = abs(left_knee[1] - right_knee[1])
    
    if torso_angle < 45:
        return "Bending"
    elif left_knee_angle < 100 or right_knee_angle < 100:
        if stride_length > 50 and knee_difference > 30:
            return "Running"
        return "Walking"
    elif left_knee_angle > 160 and right_knee_angle > 160:
        return "Standing"
    elif nose[1] > left_hip[1] and nose[1] > right_hip[1]:
        return "Lying on Floor"
    elif arm_angle > 120:
        return "Arm Raising"
    elif left_hip[1] < left_knee[1] and right_hip[1] < right_knee[1]:
        return "Jumping"
    return "Person"

# Process video if uploaded
if uploaded_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    
    cap = cv2.VideoCapture(temp_video.name)
    frame_placeholder = st.empty()
    process_video = st.button("Start Processing Video")

    if process_video:
        processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Object Detection
            object_results = object_model(frame)
            
            # Pose Estimation
            pose_results = pose_model(frame)
            
            # Draw Object Detection
            for result in object_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = object_model.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw Pose Estimation
            for result in pose_results:
                for box, keypoints in zip(result.boxes, result.keypoints.xy):
                    keypoints = [tuple(map(int, kp)) for kp in keypoints]
                    action = classify_pose(keypoints)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write to output video
            out.write(frame)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)
        
        cap.release()
        out.release()
        os.remove(temp_video.name)
        
        # Provide download link
        with open(processed_video_path, "rb") as file:
            st.download_button(label="Download Processed Video", data=file, file_name="processed_video.mp4", mime="video/mp4")
