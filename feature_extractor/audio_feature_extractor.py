import glob,os
import audio_utils
import tqdm,numpy as np

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_io as tfio

import feature_extractor_config as cfg
import audio_utils



def extract_audio_features(save = True, return_feat = False):
    
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    
    extractor_config = cfg.AUDIO_CONFIG
    audio_file_paths = [audio_utils.video_to_wav(i) for i in extractor_config.video_file_list]
    for audio_file_path in tqdm.tqdm(audio_file_paths):
        if audio_file_path != "[NO_AUDIO]":
            audio_utils.wav_to_audio_embedding(audio_file_path,yamnet_model,extractor_config, save, return_feat)
    

if __name__ == '__main__':
    extract_audio_features()
