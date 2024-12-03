import os, sys, argparse



parser = argparse.ArgumentParser(description='Choose between YouTube or video URL')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--yt_url', type=str, help='YouTube single video URL', default='')
group.add_argument('--video_url', type=str, help='Single video URL')

parser.add_argument('--source_language', type=str, help='Video source language', required=True)
parser.add_argument('--target_language', type=str, help='Video target language', required=True)
parser.add_argument('--LipSync', type=bool, help='Lip synchronization of the resut audio to the synthesized video', required=True)
parser.add_argument('--Bg_sound', type=bool, help='Keep the background sound of the original video, though it might be slightly noisy', default=False)



args = parser.parse_args()



def main():
  print(args.yt_url)
  print(args.video_url)
  print(args.source_language)
  print(args.target_language)
  print(args.LipSync)
  print(args.Bg_sound)
  