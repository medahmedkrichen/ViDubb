try:
    from pyannote.audio import Pipeline
except ModuleNotFoundError:
    Pipeline = None

try:
    from audio_separator.separator import Separator
except ModuleNotFoundError:
    Separator = None

try:
    import whisper
except ModuleNotFoundError:
    whisper = None

try:
    from transformers import MarianMTModel, MarianTokenizer
except ModuleNotFoundError:
    MarianMTModel = None
    MarianTokenizer = None

try:
    from TTS.api import TTS
except ModuleNotFoundError:
    TTS = None

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    AudioSegment = None

try:
    from speechbrain.inference.interfaces import foreign_class
except ModuleNotFoundError:
    foreign_class = None

try:
    from deepface import DeepFace
except ModuleNotFoundError:
    DeepFace = None

try:
    from groq import Groq
except ModuleNotFoundError:
    Groq = None

try:
    from IPython.display import HTML, Audio
except ModuleNotFoundError:
    HTML = None
    Audio = None

try:
    from scipy.io.wavfile import read as wav_read
except ModuleNotFoundError:
    wav_read = None

try:
    import ffmpeg
except ModuleNotFoundError:
    ffmpeg = None

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    clear_output = None

import shutil
import subprocess
try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import json
import re
from base64 import b64decode
import io
import sys, argparse
import os


def merge_overlapping_periods(period_dict):
            if not period_dict:
                return {}

            # Sort periods by start time
            sorted_periods = sorted(period_dict.items(), key=lambda x: x[0][0])

            merged_periods = []
            current_period, current_speaker = sorted_periods[0]

            for next_period, next_speaker in sorted_periods[1:]:
                # If periods overlap
                if current_period[1] >= next_period[0]:
                    # Extend the current period if they are from the same speaker
                    if current_speaker == next_speaker:
                        current_period = (current_period[0], max(current_period[1], next_period[1]))
                    # Otherwise, treat the overlap as a separate period
                    else:
                        merged_periods.append((current_period, current_speaker))
                        current_period, current_speaker = next_period, next_speaker
                else:
                    # No overlap, add the current period to the result
                    merged_periods.append((current_period, current_speaker))
                    current_period, current_speaker = next_period, next_speaker

            # Append the last period
            merged_periods.append((current_period, current_speaker))

            # Convert back to dictionary
            return dict(merged_periods)
	

def get_speaker(time_frame, speaker_dict):
                for (start, end), speaker in speaker_dict.items():
                    if start <= time_frame <= end:
                        return speaker
                return None


def extract_frames(video_path, output_folder, periods, num_frames=50):
                if cv2 is None:
                    raise ModuleNotFoundError("OpenCV is required to extract frames.")
                if not periods:
                    return

                video = cv2.VideoCapture(video_path)
                if not video.isOpened():
                    print("Error: Could not open video.")
                    return

                # Get frame rate (frames per second)
                fps = video.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    video.release()
                    print("Error: Invalid frame rate for video.")
                    return

                # Create the main folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Process each speaker period
                for (start_time, end_time), speaker in periods.items():
                    # Calculate the total number of frames for the period
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)

                    if end_frame <= start_frame:
                        continue

                    # Calculate frame intervals to pick 'num_frames' equally spaced frames
                    step = 1

                    # Create a folder for the speaker if it doesn't exist
                    speaker_folder = os.path.join(output_folder, str(speaker))
                    if not os.path.exists(speaker_folder):
                        os.makedirs(speaker_folder)

                    frame_count = 0
                    frame_number = start_frame

                    # Set the video to the start frame
                    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    while frame_number < end_frame and frame_count < num_frames:
                        success, frame = video.read()

                        if not success:
                            break

                        if frame_count % step == 0:
                            # Save the frame as an image in the speaker folder
                            frame_filename = os.path.join(speaker_folder, f"{speaker}_frame_{frame_number}.jpg")
                            cv2.imwrite(frame_filename, frame)
                            print(f"Saved frame {frame_number} for {speaker}")

                        frame_number += 1
                        frame_count += 1

                # Release the video capture object
                video.release()


def detect_and_crop_faces(image_path, face_cascade):
                if cv2 is None:
                    raise ModuleNotFoundError("OpenCV is required to detect faces.")
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Error reading image: {image_path}")
                    return False

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    return False  # No faces detected

                # Assuming we only care about the first detected face
                (x, y, w, h) = faces[0]

                # Crop the face from the image
                face = img[y:y + h, x:x + w]

                # Replace the original image with the cropped face
                cv2.imwrite(image_path, face)
                return True


def cosine_similarity(embedding1, embedding2):
                """Calculate cosine similarity between two face embeddings"""
                if np is None:
                    raise ModuleNotFoundError("NumPy is required to calculate cosine similarity.")
                embedding1 = np.asarray(embedding1, dtype=float)
                embedding2 = np.asarray(embedding2, dtype=float)
                norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                if np.isclose(norm_product, 0.0):
                    return 0.0
                return float(np.dot(embedding1, embedding2) / norm_product)


def extract_and_save_most_common_face(folder_path, threshold=0.1):
                """
                Extracts and saves the most common face from the folder, saving it as 'max_image.jpg'.
                """
                if not os.path.isdir(folder_path):
                    return None

                if DeepFace is None:
                    raise ModuleNotFoundError("DeepFace is required to extract face embeddings.")

                face_encodings = []
                face_images = {}

                # Step 1: Extract embeddings for all images in the folder
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                        file_path = os.path.join(folder_path, filename)

                        try:
                            # Get the face embedding for the image using DeepFace
                            embedding = DeepFace.represent(img_path=file_path, model_name="ArcFace")[0]["embedding"]
                            face_encodings.append(embedding)
                            face_images[tuple(embedding)] = file_path  # Store the corresponding image for the encoding
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
                            continue

                if not face_encodings:
                    return None

                # Step 3: Group faces based on similarity threshold
                unique_faces = []
                grouped_faces = {}

                for encoding in face_encodings:
                    found_match = False
                    for unique_face in unique_faces:
                        similarity = cosine_similarity(encoding, unique_face)
                        if similarity > threshold:  # If similarity is higher than the threshold, it's the same face
                            found_match = True
                            grouped_faces[tuple(unique_face)].append(encoding)  # Add current encoding to the same group
                            break
                    if not found_match:
                        unique_faces.append(encoding)
                        grouped_faces[tuple(encoding)] = [encoding]  # Start a new group for this unique face

                # Step 4: Find the most common face group
                most_common_group = max(grouped_faces, key=lambda x: len(grouped_faces[x]))

                # The image corresponding to the most common group
                most_common_image = face_images[most_common_group]

                # Step 5: Save the most common face image as "max_image.jpg"
                new_image_path = os.path.join(folder_path, "max_image.jpg")
                shutil.copy(most_common_image, new_image_path)  # Copy the image to the new path with the desired name

                print(f"Most common face extracted and saved as {new_image_path}")
                return new_image_path


def get_overlap(range1, range2):
            """Calculate the overlap between two ranges."""
            start1, end1 = range1
            start2, end2 = range2
            # Find the maximum of the start times and the minimum of the end times
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            # Calculate overlap duration
            return max(0, overlap_end - overlap_start)
