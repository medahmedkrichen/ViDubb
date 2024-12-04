import os, sys, argparse
from dotenv import load_dotenv
from audio_separator.separator import Separator
import whisper
from transformers import MarianMTModel, MarianTokenizer
import os
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



def main():
  print(args.yt_url)
  print(args.video_url)
  print(args.source_language)
  print(args.target_language)
  print(args.LipSync)
  print(args.Bg_sound)
  print('##########')
  print(os.getenv('HF_TOKEN'))
  print(os.getenv('Groq_TOKEN'))

if __name__ == '__main__':
	main()
  
