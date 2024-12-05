import cv2
import json
import re
import os
import importlib.util


def merge_overlapping_periods(period_dict):
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
                # Open the video file
                video = cv2.VideoCapture(video_path)
                
                # Get frame rate (frames per second)
                fps = video.get(cv2.CAP_PROP_FPS)
            
                if not video.isOpened():
                    print("Error: Could not open video.")
                    return
            
                # Create the main folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
            
                # Process each speaker period
                for (start_time, end_time), speaker in periods.items():
                    # Calculate the total number of frames for the period
                    start_frame = int(start_time * fps)
                    end_frame = int(end_time * fps)
                    total_frames = end_frame - start_frame
                    
                    # Calculate frame intervals to pick 'num_frames' equally spaced frames
                    step = 1
                    
                    # Create a folder for the speaker if it doesn't exist
                    speaker_folder = os.path.join(output_folder, speaker)
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
                embedding1 = np.array(embedding1)
                embedding2 = np.array(embedding2)
                return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def extract_and_save_most_common_face(folder_path, threshold=0.3):
                """
                Extracts and saves the most common face from the folder, saving it as 'max_image.jpg'.
                """
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
