import os
import importlib.util
from ascii_magic import AsciiArt

my_art = AsciiArt.from_image('Vidubb_without_bg.png')
my_art.to_terminal()


print("Start Processing...")
def install_if_not_installed(import_name, install_command):
    try:
        __import__(import_name)
    except ImportError:
        os.system(f"{install_command} > /dev/null 2>&1")

install_if_not_installed('protobuf', 'pip install protobuf==3.19.6')
install_if_not_installed('spacy', 'pip install spacy==3.8.2')
install_if_not_installed('TTS', 'pip install --no-deps TTS==0.21.0')
install_if_not_installed('packaging', 'pip install packaging==20.9')
install_if_not_installed('openai-whisper', 'pip install openai-whisper==20240930')
install_if_not_installed('deepface', 'pip install deepface==0.0.93')
os.system('pip install numpy==1.26.4 > /dev/null 2>&1')

from pyannote.audio import Pipeline
from audio_separator.separator import Separator
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from pydub import AudioSegment
import shutil
import subprocess
import torch
from speechbrain.inference.interfaces import foreign_class
from deepface import DeepFace
import numpy as np
import cv2
import json
import re
from groq import Groq
from IPython.display import HTML, Audio
from base64 import b64decode
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
from IPython.display import clear_output 
import sys, argparse
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import warnings
from tools.utils import merge_overlapping_periods
from tools.utils import get_speaker
from tools.utils import extract_frames
from tools.utils import detect_and_crop_faces
from tools.utils import cosine_similarity
from tools.utils import extract_and_save_most_common_face
from tools.utils import get_overlap
from faster_whisper import WhisperModel

        
nltk.download('punkt')
warnings.filterwarnings("ignore")
load_dotenv()

parser = argparse.ArgumentParser(description='Choose between YouTube or video URL')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--yt_url', type=str, help='YouTube single video URL', default='')
group.add_argument('--video_url', type=str, help='Single video URL')

parser.add_argument('--source_language', type=str, help='Video source language', required=True)
parser.add_argument('--target_language', type=str, help='Video target language', required=True)
parser.add_argument('--whisper_model', type=str, help='Chose the whisper model based on your device requirements', default="turbo")
parser.add_argument('--LipSync', type=bool, help='Lip synchronization of the resut audio to the synthesized video', default=False)
parser.add_argument('--Bg_sound', type=bool, help='Keep the background sound of the original video, though it might be slightly noisy', default=False)



args = parser.parse_args()



class VideoDubbing:
    def __init__(self, Video_path, source_language, target_language, 
                 LipSync=True, Voice_denoising = True, whisper_model="turbo",
                 Context_translation = "API code here", huggingface_auth_token="API code here"):
        
        self.Video_path = Video_path
        self.source_language = source_language
        self.target_language = target_language
        self.LipSync = LipSync
        self.Voice_denoising = Voice_denoising
        self.whisper_model = whisper_model
        self.Context_translation = Context_translation
        self.huggingface_auth_token = huggingface_auth_token
        

        os.system("rm -r audio")
        os.system("mkdir audio")


        os.system("rm -r results")
        os.system("mkdir results")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the pre-trained speaker diarization pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                     use_auth_token=self.huggingface_auth_token).to(device)
        
        # Load the audio from the video file
        audio = AudioSegment.from_file(self.Video_path, format="mp4")
        audio.export("audio/test0.wav", format="wav")
        
        
        audio_file = "audio/test0.wav"
        
        # Apply the diarization pipeline on the audio file
        diarization = pipeline(audio_file)
        speakers_rolls ={}
        
        # Print the diarization results
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            if abs(speech_turn.end - speech_turn.start) > 1.5:
                print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
                speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker
        
        
        
        
        # speakers_rolls = merge_overlapping_periods(speakers_rolls)

        if self.LipSync:
            # Load the video file
            video = cv2.VideoCapture(self.Video_path)
            
            # Get frames per second (FPS)
            fps = video.get(cv2.CAP_PROP_FPS)
            
            # Get total number of frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video.release()
            
            
            
            
            frame_per_speaker = []
            
            for i in range(total_frames):
                time = i/round(fps)
                frame_speaker = get_speaker(time, speakers_rolls)
                frame_per_speaker.append(frame_speaker)
                # print(time)
            
            os.system("rm -r speakers_image")
            os.system("mkdir speakers_image")
            
            
            
            # Specify the video path and output folder
            output_folder = "speakers_image"
            # Call the function
            extract_frames(self.Video_path, output_folder, speakers_rolls)
            
            # Initialize the MTCNN face detector
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            # Load the pre-trained Haar Cascade model for face detection
            face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            
            # Function to detect and crop faces
            
            # Path to the folder containing speaker images
            speaker_images_folder = "speakers_image"
            
            # Iterate through speaker subfolders
            for speaker_folder in os.listdir(speaker_images_folder):
                speaker_folder_path = os.path.join(speaker_images_folder, speaker_folder)
            
                if os.path.isdir(speaker_folder_path):
                    # Process each image in the speaker folder
                    for image_name in os.listdir(speaker_folder_path):
                        image_path = os.path.join(speaker_folder_path, image_name)
            
                        # Detect and crop faces from the image
                        if not detect_and_crop_faces(image_path, face_cascade):
                            # If no face is detected, delete the image
                            os.remove(image_path)
                            print(f"Deleted {image_path} due to no face detected.")
                        else:
                            print(f"Face detected and cropped: {image_path}")
            
            
        
            
            speaker_images_folder = "speakers_image"
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
            
            
            if os.path.exists("Wav2Lip/speakers_image"):
                shutil.rmtree("Wav2Lip/speakers_image")
            shutil.copytree("speakers_image", "Wav2Lip/speakers_image")
            

            
        ###############################################################################
        
        os.system("rm -r speakers_audio")
        os.system("mkdir speakers_audio")
        
        speakers = set(list(speakers_rolls.values()))
        audio = AudioSegment.from_file(audio_file, format="mp4")
        
        for speaker in speakers:
            speaker_audio = AudioSegment.empty()
            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(key[0])*1000
                    end = int(key[1])*1000
                    
                    speaker_audio += audio[start:end]
                    
        
            speaker_audio.export(f"speakers_audio/{speaker}.wav", format="wav")
        
        most_occured_speaker= max(list(speakers_rolls.values()),key=list(speakers_rolls.values()).count)
        
        model = WhisperModel("turbo", device="cuda")
        segments, info = model.transcribe("/kaggle/input/Not Your Average 18 Year Old - Donald Trump OPENS UP About Baron Trump - Valuetainment (720p h264 youtube).mp4", word_timestamps=True)
        segments = list(segments) 
			 
        time_stamped = []
        full_text = []
        for segment in segments:
                for word in segment.words:
                        time_stamped.append([word.word, word.start, word.end])
                        full_text.append(word.word)
        full_text = "".join(full_text)       
        # Decompose Long Sentences

        
        
        # Tokenize the text into sentences
        tokenized_sentences = sent_tokenize(full_text)
        sentences = []
        
        # Print the sentences
        for i, sentence in enumerate(tokenized_sentences):
            sentences.append(sentence)

        
        time_stamped_sentances = {}
        count_sentances = {}
        
        letter = 0
        for i in range(len(sentences)):
            tmp = []
            starts = []
            
            for j in range(len(sentences[i])):
                letter += 1
                tmp.append(sentences[i][j])
                
                f = 0
                for k in range(len(time_stamped)):
                    for m in range(len(time_stamped[k][0])):
                        f += 1
                        
                        if f == letter:
        
                            starts.append(time_stamped[k][1])
                       
                            starts.append(time_stamped[k][2])
            letter += 1               
                            
            time_stamped_sentances["".join(tmp)] = [min(starts), max(starts)]
            count_sentances[i+1] = "".join(tmp)

        record = []
        print(time_stamped_sentances)
        for sentence in time_stamped_sentances:
            record.append([sentence, time_stamped_sentances[sentence][0], time_stamped_sentances[sentence][1]])
        
       
        
       
        # Decompose Long Sentences
        
        """record = []
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
            i += 1"""
        
        new_record = record
        
        # Audio Emotions Analysis
        
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={"device":f"{device}"})
        
        emotion_dict = {'neu': 'Neutral',
                        'ang' : 'Angry',
                        'hap' : 'Happy',
                        'sad' : 'Sad',
                        'None': None}
    

        
        if not self.Context_translation:

            # Function to translate text
            def translate(sentence):
                if self.source_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-trk-{self.target_language}"
                elif self.target_language == 'tr':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-trk"
                elif self.source_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-zh-{self.target_language}"
                elif self.target_language == 'zh-cn':
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-zh"
                else:
                    model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"
	
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(device)
                model = model.to('cpu')
                inputs = tokenizer([sentence], return_tensors="pt", padding=True).to('cpu')
                translated = model.generate(**inputs)
                return tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            client = Groq(api_key=self.Context_translation)

            def translate(sentence, before_context, after_context, target_language):
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

                try:
                    translation = match.group(1)
                    return translation
                except:
                    return 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                    
               

        
        records = []
        
        audio = AudioSegment.from_file(audio_file, format="mp4")
        for i in range(len(new_record)):
            final_sentance = new_record[i][0]
            if not self.Context_translation:
                translated = translate(sentence=final_sentance)
                
            else:
                before_context = new_record[i-1][0] if i - 1 in range(len(new_record)) else ""
                after_context = new_record[i+1][0] if i + 1 in range(len(new_record)) else ""
                translated = translate(sentence=final_sentance, before_context=before_context, after_context=after_context, target_language=self.target_language )
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
                audio[start:end].export("audio/emotions.wav", format="wav")      
                out_prob, score, index, text_lab = classifier.classify_file("audio/emotions.wav")
                os.remove("audio/emotions.wav")
            except:
                text_lab = ['None']
            
            records.append([translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]]])
            print(translated, final_sentance, new_record[i][1], new_record[i][2], speaker, emotion_dict[text_lab[0]])
        
        
        
        os.environ["COQUI_TOS_AGREED"] = "1"
        if device == "cuda":
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        else:
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        #!tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  --list_speaker_idxs
        
        os.system("rm -r audio_chunks")
        os.system("rm -r su_audio_chunks")
        os.system("mkdir audio_chunks")
        os.system("mkdir su_audio_chunks")

        natural_scilence = records[0][2]
        previous_silence_time = 0
        
        if natural_scilence >= 0.8:
            previous_silence_time = 0.8
            natural_scilence -= 0.8
        else:
            previous_silence_time = natural_scilence
            natural_scilence = 0   
            
        combined = AudioSegment.silent(duration=natural_scilence*1000) 

        tip = 350
        
        for i in range(len(records)):
            print('previous_silence_time: ', previous_silence_time)
            tts.tts_to_file(text=records[i][0],
                        file_path=f"audio_chunks/{i}.wav",
                        speaker_wav=f"speakers_audio/{records[i][4]}.wav",
                        language=self.target_language,
                        emotion=records[i][5],
                        speed=2)
            
            audio = AudioSegment.from_file(f"audio_chunks/{i}.wav")
            audio = audio[:len(audio)-tip]
            audio.export(f"audio_chunks/{i}.wav", format="wav")
            
            
            lt = len(audio) / 1000.0 
            lo =  max(records[i][3] - records[i][2], 0)
            theta = lo/lt
          
            input_file = f"audio_chunks/{i}.wav"
            output_file = f"su_audio_chunks/{i}.wav"

           
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
            del audio
        
       
        
        # Get all the audio files from the folder
        audio_files = [f for f in os.listdir("su_audio_chunks") if f.endswith(('.mp3', '.wav', '.ogg'))]
        
        # Sort files to concatenate them in order, if necessary
        audio_files.sort(key=lambda x: int(x.split('.')[0]))  # Modify sorting logic if needed (e.g., based on filenames)
        
        # Loop through and concatenate each audio file
        for audio_file in audio_files:
            file_path = os.path.join("su_audio_chunks", audio_file)
            audio_segment = AudioSegment.from_file(file_path)
            combined += audio_segment  # Append audio to the combined segment
        
        
        audio = AudioSegment.from_file(self.Video_path)
        total_length = len(audio) / 1000.0 
        silence = AudioSegment.silent(duration=abs(total_length - records[-1][3])*1000)
        combined += silence
        # Export the combined audio to the output file
        combined.export("audio/output.wav", format="wav")

        
        # Initialize Spleeter with the 2stems model (vocals + accompaniment)
        separator = Separator()

        # Load a model
        separator.load_model(model_filename='2_HP-UVR.pth')
        output_file_paths = separator.separate(self.Video_path)[0]

      
        
        
        audio1 = AudioSegment.from_file("audio/output.wav")
        audio2 = AudioSegment.from_file(output_file_paths)
        combined_audio = audio1.overlay(audio2)
        
        # Export the combined audio file
        combined_audio.export("audio/combined_audio.wav", format="wav")
        
        
        # Video and Audio Overlay
        
        command = f"ffmpeg -i '{self.Video_path}' -i audio/combined_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest output_video.mp4"
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        shutil.move(output_file_paths, "audio/")
        
        
        
        if self.Voice_denoising:
            
            """model, df_state, _ = init_df()
            audio, _ = load_audio("audio/combined_audio.wav", sr=df_state.sr())
            # Denoise the audio
            enhanced = enhance(model, df_state, audio)
            # Save for listening
            save_audio("audio/enhanced.wav", enhanced, df_state.sr())"""
            command = f"ffmpeg -i '{self.Video_path}' -i audio/output.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest denoised_video.mp4"
            subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if self.LipSync and self.Voice_denoising:
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../denoised_video.mp4' --audio '../audio/output.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")
            
        if self.LipSync and not self.Voice_denoising:
            os.system("pip install librosa==0.9.1 > /dev/null 2>&1")
            os.system("cd Wav2Lip && python inference.py --checkpoint_path 'wav2lip_gan.pth' --face '../output_video.mp4' --audio '../audio/combined_audio.wav' --face_det_batch_size 1 --wav2lip_batch_size 1")

			 
        if  self.LipSync and self.Voice_denoising:
            source_path = 'Wav2Lip/results/result_voice.mp4'
            destination_folder = 'results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
            shutil.move('denoised_video.mp4', destination_folder)

        elif self.LipSync and not self.Voice_denoising:
            source_path = 'Wav2Lip/results/result_voice.mp4'
            destination_folder = 'results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
            os.remove('denoised_video.mp4')
		
        elif not self.LipSync and self.Voice_denoising:
            source_path = 'denoised_video.mp4'
            destination_folder = 'results'

            shutil.move(source_path, destination_folder)
            os.remove('output_video.mp4')
        else:
            source_path = 'output_video.mp4'
            destination_folder = 'results'

            shutil.move(source_path, destination_folder)
	
        os.system('pip install -r requirements.txt > /dev/null 2>&1')	

def main():
	os.system("rm video_path.mp4")
	video_path = None
	if args.yt_url:
		os.system(f"yt-dlp -f best -o 'video_path.mp4' --recode-video mp4 {args.yt_url}")
		video_path = "video_path.mp4"

	if not video_path:
		video_path = args.video_url
	
	vidubb = VideoDubbing(video_path, args.source_language, args.target_language, args.LipSync, not args.Bg_sound, args.whisper_model, os.getenv('Groq_TOKEN'), os.getenv('HF_TOKEN'))
	
if __name__ == '__main__':
	main()
  
