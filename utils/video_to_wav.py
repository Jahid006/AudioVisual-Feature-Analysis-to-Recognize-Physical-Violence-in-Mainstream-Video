import tqdm,numpy as np,os,tqdm,glob
import moviepy.editor as mpy

start = 0
#class_ = "PhysicalViolence"
data  = glob.glob(r'H:\S-Home\Home Office\Violence\archive\Real Life Violence Dataset\*\*', recursive=True)


def video_to_wav(video):
    """convert video to wav using moviepy and save it in audio directory"""
    clip = mpy.VideoFileClip(video)
    audio = clip.audio
    path = video[:-4] + ".wav"
    audio.write_audiofile(path)
    
    
print(len(data))
for i in tqdm.tqdm(data):
    video_to_wav(i)
    
    
    
'''
import tqdm,numpy as np,os,tqdm,glob
import moviepy.editor as mpy
from IPython.display import clear_output

st = 0
class_ = "PhysicalViolence"
data  = glob.glob(f'dataset\{class_}\*')


def video_to_wav(video):
    #"""convert video to wav using moviepy and save it in audio directory"""
    clip = mpy.VideoFileClip(video)
    audio = clip.audio
    path = f'audio/{class_}/' + video.split('\\')[-1][:-4] + '.wav'
    audio.write_audiofile(path)
    print(path)
    return path

os.makedirs(f"audio/{class_}" , exist_ok=True)
#os.makedirs(f"audio_embedding/{class_}" , exist_ok=True)
print("Data: ",data, len(data))

for i in tqdm.tqdm(data):
    video_to_wav(i)
    clear_output()
    
'''