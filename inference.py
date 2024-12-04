
import os
import importlib.util

print("Start Processing...")
def install_if_not_installed(import_name, install_command):
    try:
        __import__(import_name)
    except ImportError:
        os.system(f"{install_command}")

install_if_not_installed('TTS', 'pip install TTS==0.22.0')

install_if_not_installed('pyannote.audio', 'pip install pyannote.audio==3.3.2')

install_if_not_installed('deepface', 'pip install deepface==0.0.93')
install_if_not_installed('librosa', 'pip install librosa==0.9.1')

install_if_not_installed('packaging', 'pip install packaging==20.9')

from IPython.display import HTML, Audio
from base64 import b64decode
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
from IPython.display import clear_output 
import sys, argparse
from dotenv import load_dotenv
from audio_separator.separator import Separator
import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from pydub import AudioSegment
import shutil
import subprocess
from pyannote.audio import Pipeline
import torch
from speechbrain.inference.interfaces import foreign_class
from deepface import DeepFace
import numpy as np
import cv2
import json
import re
from groq import Groq


load_dotenv()

parser = argparse.ArgumentParser(description='Choose between YouTube or video URL')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--yt_url', type=str, help='YouTube single video URL', default='')
group.add_argument('--video_url', type=str, help='Single video URL')

parser.add_argument('--source_language', type=str, help='Video source language', required=True)
parser.add_argument('--target_language', type=str, help='Video target language', required=True)
parser.add_argument('--LipSync', type=bool, help='Lip synchronization of the resut audio to the synthesized video', default=False)
parser.add_argument('--Bg_sound', type=bool, help='Keep the background sound of the original video, though it might be slightly noisy', default=False)



args = parser.parse_args()
class VideoDubbing:
    def __init__(self, Video_path, source_language, target_language, 
                 LipSync=True, Voice_denoising = True, 
                 Context_translation = "API code here", huggingface_auth_token="API code here"):
        
        self.Video_path = Video_path
        self.source_language = source_language
        self.target_language = target_language
        self.LipSync = LipSync
        self.Voice_denoising = Voice_denoising
        self.Context_translation = Context_translation
        self.huggingface_auth_token = huggingface_auth_token
        
        # Speaker Diarization

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the pre-trained speaker diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                     use_auth_token=self.huggingface_auth_token).to(device)
        
        # Load the audio from the video file
        audio = AudioSegment.from_file(self.Video_path, format="mp4")
        audio.export("test0.wav", format="wav")
        
        
        audio_file = "test0.wav"
        
        # Apply the diarization pipeline on the audio file
        diarization = pipeline(audio_file)
        speakers_rolls ={}
        
        # Print the diarization results
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
            speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
        
        
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
        
        speakers_rolls = merge_overlapping_periods(speakers_rolls)

        if self.LipSync:
            # Load the video file
            video = cv2.VideoCapture(self.Video_path)
            
            # Get frames per second (FPS)
            fps = video.get(cv2.CAP_PROP_FPS)
            
            # Get total number of frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video.release()
            
            
            def get_speaker(time_frame, speaker_dict):
                for (start, end), speaker in speaker_dict.items():
                    if start <= time_frame <= end:
                        return speaker
                return None
            
            frame_per_speaker = []
            
            for i in range(total_frames):
                time = i/round(fps)
                frame_speaker = get_speaker(time, speakers_rolls)
                frame_per_speaker.append(frame_speaker)
                # print(time)
            
            os.system("rm -r speaker_images")
            os.system("mkdir speaker_images")
            
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
            
            # Specify the video path and output folder
            output_folder = 'speaker_images'
            # Call the function
            extract_frames(self.Video_path, output_folder, speakers_rolls)
            
            # Initialize the MTCNN face detector
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            # Load the pre-trained Haar Cascade model for face detection
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            
            # Function to detect and crop faces
            def detect_and_crop_faces(image_path):
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
            
            # Path to the folder containing speaker images
            speaker_images_folder = "speaker_images"
            
            # Iterate through speaker subfolders
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(speaker_images_folder, speaker_folder)
            
                if os.path.isdir(speaker_folder_path):
                    # Process each image in the speaker folder
                    for image_name in os.listdir(speaker_folder_path):
                        image_path = os.path.join(speaker_folder_path, image_name)
            
                        # Detect and crop faces from the image
                        if not detect_and_crop_faces(image_path):
                            # If no face is detected, delete the image
                            os.remove(image_path)
                            print(f"Deleted {image_path} due to no face detected.")
                        else:
                            print(f"Face detected and cropped: {image_path}")
            
            
            
            # Step 2: Compare face embeddings with a threshold for similarity
            def cosine_similarity(embedding1, embedding2):
                """Calculate cosine similarity between two face embeddings"""
                embedding1 = np.array(embedding1)
                embedding2 = np.array(embedding2)
                return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            
            def extract_and_save_most_common_face(folder_path, threshold=0.4):
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
            
            speaker_images_folder = "speaker_images"
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(speaker_images_folder, speaker_folder)
            
                print(f"Processing images in folder: {speaker_folder}")
                extract_and_save_most_common_face(speaker_folder_path)

            for root, dirs, files in os.walk(speaker_images_folder):
                for file in files:
                    # Check if the file is not 'max_image.jpg'
                    if file != "max_image.jpg":
                        # Construct full file path
                        file_path = os.path.join(root, file)
                        # Delete the file
                        os.remove(file_path)
            
            
            
            # Save to a file
            with open('frame_per_speaker.json', 'w') as f:
                json.dump(frame_per_speaker, f)
            
            
            if os.path.exists("Wav2Lip/frame_per_speaker.json"):
                os.remove("Wav2Lip/frame_per_speaker.json")
            shutil.copyfile('frame_per_speaker.json', "Wav2Lip/frame_per_speaker.json")
            
            
            if os.path.exists("Wav2Lip/speaker_images"):
                shutil.rmtree("Wav2Lip/speaker_images")
            shutil.copytree("speaker_images", "Wav2Lip/speaker_images")
            

            
        ###############################################################################
        
        os.system("rm -r speakers")
        os.system("mkdir speakers")
        
        speakers = set(list(speakers_rolls.values()))
        audio = AudioSegment.from_file(audio_file, format="mp4")
        
        for speaker in speakers:
            speaker_audio = AudioSegment.empty()
            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(key[0])*1000
                    end = int(key[1])*1000
                    
                    speaker_audio += audio[start:end]
                    
        
            speaker_audio.export(f"speakers/{speaker}.wav", format="wav")
        
        most_occured_speaker= max(list(speakers_rolls.values()),key=list(speakers_rolls.values()).count)
        
        model = whisper.load_model("large-v3", device=device)
        transcript = model.transcribe(
            word_timestamps=True,
            audio=self.Video_path,
          )
        for segment in transcript['segments']:
            print(''.join(f"{word['word']}[{word['start']}/{word['end']}]"
                            for word in segment['words']))
        
       
        
       
        # Decompose Long Sentences
        
        record = []
        for segment in transcript['segments']:
            print("#############################")
            sentance = []
            starts = []
            ends = []
            i = 1
            if len(segment['text'].split())>25:
                k = len(segment['text'].split())//4
            else:
                k = 25
            for word in segment['words']:
                if i % k != 0:
                    i += 1
                    sentance.append(word['word'])
                    starts.append(word['start'])
                    ends.append(word['end'])
                    
                else:
                     i += 1
                     final_sentance = " ".join(sentance)
                     if starts and ends and final_sentance:
                         print(final_sentance+f'[{min(starts)} / {max(ends)}]')
                         record.append([final_sentance, min(starts), max(ends)])
                      
                     sentance = []
                     starts = []
                     ends = []
            final_sentance = " ".join(sentance)         
            if starts and ends and final_sentance:
                print(final_sentance+f'[{min(starts)} / {max(ends)}]')
                record.append([final_sentance, min(starts), max(ends)])
                sentance = []
                starts = []
                ends = []
        
        i = 1
        new_record = [record[0]]
        while i <len(record)-1:
            if len(new_record[-1][0].split()) +  len(record[i][0].split()) < 10:
                text = new_record[-1][0]+record[i][0]
                start = new_record[-1][1]
                end = record[i][2]
                del new_record[-1]
                new_record.append([text, start, end])
            else:
                new_record.append(record[i])
            i += 1
        
        new_record = record
        
        # Audio Emotions Analysis
        
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={"device":"cuda"})
        
        emotion_dict = {'neu': 'Neutral',
                        'ang' : 'Angry',
                        'hap' : 'Happy',
                        'sad' : 'Sad',
                        'None': None}
    
        def get_overlap(range1, range2):
            """Calculate the overlap between two ranges."""
            start1, end1 = range1
            start2, end2 = range2
            # Find the maximum of the start times and the minimum of the end times
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            # Calculate overlap duration
            return max(0, overlap_end - overlap_start)


         # 
        model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)

        if not self.Context_translation:
        
           
            
            # Function to translate text
            def translate(sentence, model, tokenizer):
                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to(device)
                translated = model.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            client = Groq(api_key=self.Context_translation)

            def translate(sentence, before_context, after_context, target_language, model, tokenizer):
                chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Role: You are a professional translator who translates concisely in short sentence while preserving meaning.
                        Instruction:
                        Translate the given sentence into {target_language}
                        
                      
                        Sentence: {sentence}
                
                        
                        Output format:
                        [[sentence translation: <your translation>]]
                        """,
                    }
                ],
                model="llama3-70b-8192",
            )
            # return chat_completion.choices[0].message.content
                # Regex pattern to extract the translation
                pattern = r'\[\[sentence translation: (.*?)\]\]'
                
                # Extracting the translation
                match = re.search(pattern, chat_completion.choices[0].message.content)
                if match:
                    translation = match.group(1)
                    return translation
                    
                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to(device)
                translated = model.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)


        
        records = []
        
        audio = AudioSegment.from_file(audio_file, format="mp4")
        for i in range(len(new_record)):
            final_sentance = new_record[i][0]
            if not self.Context_translation:
                translated = translate(sentence=final_sentance, model=model, tokenizer=tokenizer)
                
            else:
                before_context = new_record[i-1][0] if i - 1 in range(len(new_record)) else ""
                after_context = new_record[i+1][0] if i + 1 in range(len(new_record)) else ""
                translated = translate(sentence=final_sentance, before_context=before_context, after_context=after_context, target_language=self.target_language, model=model, tokenizer=tokenizer)
            speaker = most_occured_speaker
            
            max_overlap = 0
        
            # Check overlap with each speaker's time range
            for key, value in speakers_rolls.items():
                speaker_start =  int(key[0])
                speaker_end = int(key[1])
                
                # Calculate overlap
                overlap = get_overlap((new_record[i][1], new_record[i][2]), (speaker_start, speaker_end))
                
                # Update speaker if this overlap is greater than previous ones
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = value
                    
            start = int(new_record[i][1]) *1000
            end = int(new_record[i][2]) *1000
        
            try:
                audio[start:end].export("emotions.wav", format="wav")      
                out_prob, score, index, text_lab = classifier.classify_file("emotions.wav")
                os.remove("emotions.wav")
            except:
                text_lab = ['None']
            
            records.append([translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]]])
            print(translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]])
        
        
        
        os.environ["COQUI_TOS_AGREED"] = "1"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        #!tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  --list_speaker_idxs
        
        os.system("rm -r outputs")
        os.system("rm -r outputs2")
        os.system("mkdir outputs")
        os.system("mkdir outputs2")

        previous_silence_time = 0

        for i in range(len(records)):
            print('previous_silence_time: ', previous_silence_time)
            tts.tts_to_file(text=records[i][0],
                        file_path=f"outputs/{i}.wav",
                        speaker_wav=f"speakers/{records[i][4]}.wav",
                        language=self.target_language,
                        emotion=records[i][5],
                        speed=1
                        )
            
            audio = AudioSegment.from_file(f"outputs/{i}.wav")
            lt = len(audio) / 1000.0 
            lo =  max(records[i][3] - records[i][2], 0)
            theta = lo/lt
          
            input_file = f"outputs/{i}.wav"
            output_file = f"outputs2/{i}.wav"

           
            if theta <1 and theta > 0.44:
                print('############################')
                theta_prim = (lo+previous_silence_time)/lt
                command = f"ffmpeg -i {input_file} -filter:a 'atempo={1/theta_prim}' -vn {output_file}"
                process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if process.returncode != 0:
                    sc = lo  + previous_silence_time
                    silence = AudioSegment.silent(duration=(sc*1000))
                    silence.export(output_file, format="wav")
            elif theta < 0.44:
                silence = AudioSegment.silent(duration=((lo+previous_silence_time)*1000))
                silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(previous_silence_time*1000))
                audio = silence  + audio
                audio.export(output_file, format="wav")
        
                
            audio = AudioSegment.from_file(output_file)
            lt = len(audio) / 1000.0
            lo =  records[i][3]-records[i][2]+ previous_silence_time
            if i+1 < len(records):
                natural_scilence = max(records[i+1][2]-records[i][3], 0) 
                if natural_scilence >= 0.8:
                    previous_silence_time = 0.8
                    natural_scilence -= 0.8
                else:
                    previous_silence_time = natural_scilence
                    natural_scilence = 0
                
                    
                silence = AudioSegment.silent(duration=((max(lo-lt,0)+natural_scilence)*1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(max(lo-lt,0)*1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            
            print("#######diff######: ",lo-lt)
            print("lo: ", lo)
            print("lt: ", lt)
            
        
        # Combine Audios
        
        combined = AudioSegment.silent(duration=records[0][2]*1000)
        
        # Get all the audio files from the folder
        audio_files = [f for f in os.listdir("outputs2") if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        # Sort files to concatenate them in order, if necessary
        audio_files.sort(key=lambda x: int(x.split('.')[0]))  # Modify sorting logic if needed (e.g., based on filenames)
        
        # Loop through and concatenate each audio file
        for audio_file in audio_files:
            file_path = os.path.join("outputs2", audio_file)
            audio_segment = AudioSegment.from_file(file_path)
            combined += audio_segment  # Append audio to the combined segment
        
        
        audio = AudioSegment.from_file(self.Video_path)
        total_length = len(audio) / 1000.0 
        silence = AudioSegment.silent(duration=abs(total_length - records[-1][3])*1000)
        combined += silence
        # Export the combined audio to the output file
        combined.export("output.wav", format="wav")
                
        
        # Initialize Spleeter with the 2stems model (vocals + accompaniment)
        separator = Separator()

        # Load a model
        separator.load_model(model_filename='1_HP-UVR.pth')
        output_file_paths = separator.separate(self.Video_path)[0]

      
        
        
        audio1 = AudioSegment.from_file("output.wav")
        audio2 = AudioSegment.from_file(output_file_paths)
        combined_audio = audio1.overlay(audio2)
        
        # Export the combined audio file
        combined_audio.export("combined_audio.wav", format="wav")
        
        
        # Video and Audio Overlay
        
        command = f"ffmpeg -i '{self.Video_path}' -i combined_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest output_video.mp4"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
	
        clear_output()
        
        
        if self.Voice_denoising:
            
            """model, df_state, _ = init_df()
            audio, _ = load_audio("combined_audio.wav", sr=df_state.sr())
            # Denoise the audio
            enhanced = enhance(model, df_state, audio)
            # Save for listening
            save_audio("enhanced.wav", enhanced, df_state.sr())"""
            command = f"ffmpeg -i '{self.Video_path}' -i output.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest denoised_video.mp4"
            subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if self.LipSync and self.Voice_denoising:
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../denoised_video.mp4' --audio '../output.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")
            
        if self.LipSync and not self.Voice_denoising:
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../output_video.mp4' --audio '../combined_audio.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")
def main():
	if youtube_link:
		os.system(f"yt-dlp -f best -o 'video_path.mp4' --recode-video mp4 {youtube_link}")
		video_path = "video_path.mp4"

	if not video_path:
		video_path = args.video_url
	
	print(args.yt_url)
	print(args.video_url)
	print(args.source_language)
	print(args.target_language)
	print(args.LipSync)
	print(args.Bg_sound)
	print('##########')
	print(os.getenv('HF_TOKEN'))
	print(os.getenv('Groq_TOKEN'))
	vidubb = VideoDubbing(video_path, args.source_language, args.target_language, args.LipSync, not args.Bg_sound, os.getenv('Groq_TOKEN'), os.getenv('HF_TOKEN'))
	
if __name__ == '__main__':
	main()
  
