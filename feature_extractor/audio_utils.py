import tqdm,numpy as np,os,tqdm,glob
import moviepy.editor as mpy

import tensorflow_hub as hub,tensorflow as tf
import tensorflow_io as tfio
import tqdm,numpy as np,os,tqdm,glob


def video_to_wav(video):
    """convert video to wav using moviepy and save it in audio directory"""
    path = video[:-4] + ".wav"
    #path = path.replace('video','audio')
    if os.path.exists(path):
        return path
    try:
        clip = mpy.VideoFileClip(video)
        audio = clip.audio
        
        #os.makedirs(path, exist_ok=True)
        audio.write_audiofile(path)
    except:
        return "[NO_AUDIO]"
    return path
    
@tf.function
def load_wav_16k_mono(filename):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000) if sample_rate != 16000 else wav
    
    return wav

def wav_to_audio_embedding(audio_path,yamnet_model,cfg,save= True, return_feat= False):
    """ save the embedding of the audio file """
      
    if save: name = cfg.get_save_dir(audio_path, cfg.model_name,cfg.num_segments, cfg.version, cfg.to_replace)
    if os.path.exists(name):return
    
    gg, embeddings, kk = yamnet_model(load_wav_16k_mono(audio_path))
    embeddings = np.max(embeddings,axis=0)*.25+np.mean(embeddings,axis=0)*.75 
    embeddings = embeddings.reshape(-1)
    
    if save : np.save(name,embeddings , allow_pickle=True)
    
    

def main(audio_data):
    for i in tqdm.tqdm(audio_data[:]):
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
        wav_to_audio_embedding(i,yamnet_model)
        
if __name__ == '__main__':
    audio_data = glob.glob('video/*.mp4')
    main(audio_data)