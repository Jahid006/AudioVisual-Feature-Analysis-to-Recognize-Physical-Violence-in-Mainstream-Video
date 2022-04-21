import numpy as np
import  os, joblib, json

import pims, gc
from decord import VideoReader,  cpu

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import moviepy.editor as mpy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import config as cfg

    
def times():
    now = datetime.now()
    return str(now.strftime("%d-%m-%Y..%H.%M.%S"))

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    else:
        ask_for_confirmation('Directories already exists. Proceeding to train a model with\
            this configuration might conflict with existing trained model.')  
        
def make_dirs():
    _ = [make_dir(i) for i in [cfg.HISTORY_PATH, cfg.LOG_PATH, cfg.MODEL_PATH, cfg.RESULT_PATH]]

def save(name,data,about="Undocumented"):
    
    '''This function saves file as sav with Time extension'''
    about += " at "+times()
    filename = name +"_"+ times()+'.sav' 
    joblib.dump([about,data], filename)
    print('Successfully Pickled->',name)
    
def Load(name):
    about,data = joblib.load(name)
    print("About "+name+": \n"+about)
    return data

def shuffle_index(l,seed=cfg.SEED):
    permutation  = np.random.RandomState(seed=seed).permutation((l))
    inversePermutation = np.argsort(permutation)
    return permutation, inversePermutation

def dict_to_json(data, path):
    try:
        print('Saving file as pandas dataframe.')
        data = pd.DataFrame(data) 
        with open(path, mode='w') as f:
            data.to_json(f)
        
    except Exception as e:
        ask_for_confirmation(f"{e} occured. Will save the file as serialized json.")
        with open(path, 'w') as f:
            json.dump(data, f)
               
def json_to_dict(path):
    with open(path, 'r') as f:
        return json.load(f)[0]

def plot_loss(history, title=''):
    plt.figure(figsize=(10,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{cfg.RESULT_PATH}\loss_{title}.png')

def plot_acc(history,title=''):
    plt.figure(figsize=(10,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{cfg.RESULT_PATH}\\accuracy_{title}.png')
    
def ask_for_confirmation(msg=''):
    if cfg.CHECK_CONFIRMATION:
        if(input(f'\n{msg} continue...y/n: ').lower()!='y'):
            if (input('We are exiting....y/n: ').lower()=='y'):
                exit()
    else: print(f'\n{msg}')

def generate_report(a,b,v=0): #actual,predicted
    
    a = np.array(a).astype(int)
    b = np.array(b).astype(int)
    if v: 
        a = 1-a
        b = 1-b

    tp = sum((a==0) & (b==0))
    tn = sum((a==1) & (b==1))
    fn = sum((a==0) & (b==1))
    fp = sum((a==1) & (b==0))
    
    confusion_matrix = {'TP': int(tp), 'TN': int(tn), 'FN': int(fn), 'FP': int(fp)}
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn) 
    f1_score = (2*precision*recall)/ (precision+recall)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    support = sum(a==v)
    accuracy ,precision,recall,f1_score,support = np.round([accuracy ,precision,recall,f1_score,support],3)
    
    report = {'class':v,
              'No of test samples':len(a),
              'support': int(support),
              'confusion_matrix':confusion_matrix,
              'accuracy':accuracy,
              'precision':precision,
              'recall':recall,
              'f1_score':f1_score
              }
    
    return report   

def generate_report_skl(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    confusion_matrix_ = {'TP': int(tp), 'TN': int(tn), 'FN': int(fn), 'FP': int(fp)}
    
    report = {'classification_report':classification_report(y_true, y_pred, output_dict=True),
              'No of test samples':len(y_true),
              "confusion_matrix": confusion_matrix_}
    return report
    
def video_to_wav(video_path):
    """convert video to wav using moviepy and save it in audio directory"""
    
    path = video_path[:-4] + ".wav"
    if os.path.exists(path):return 
    
    clip = mpy.VideoFileClip(video_path)
    audio = clip.audio
    
    audio.write_audiofile(path)
    
def generate_audio_embedding_using_librosa(audio_path):
    """generate audio embedding using librosa"""
    import librosa
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc

def generate_audio_embedding(audio_path):
    
    if os.path.exists(audio_path.replace('.wav','.npy')): 
        return np.load(audio_path.replace('.wav','.npy'),allow_pickle=True)
    
    else:
    
        import tensorflow_hub as hub,tensorflow as tf
        import tensorflow_io as tfio
        
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)

        @tf.function
        def load_wav_16k_mono(filename):
            """ read in a waveform file and convert to 16 kHz mono """
            file_contents = tf.io.read_file(filename)
            wav, sample_rate = tf.audio.decode_wav(
                file_contents,
                desired_channels=1)
            wav = tf.squeeze(wav, axis=-1)
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
            return wav
        
        def save_embedding(audio_path):
            """ save the embedding of the audio file """
            gg, embeddings, kk = yamnet_model(load_wav_16k_mono(audio_path))  
            embeddings = np.max(embeddings,axis=0)*.25+np.mean(embeddings,axis=0)*.75 
            embeddings = embeddings.reshape(-1)

            np.save(audio_path.replace('.wav','.npy'),embeddings , allow_pickle=True)
            
            return embeddings
        
        try : 
            return save_embedding(audio_path)
        except: 
            print(f'\n{audio_path} is not a valid audio file')
            return np.zeros(1024).flatten()
              
def get_backbone(backbone):
    import tensorflow as tf
    if backbone == 'DenseNet201':
        from tensorflow.keras.applications.densenet import preprocess_input
        backbone = tf.keras.applications.DenseNet201(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        backbone = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",pooling='avg')
        
    return backbone, preprocess_input
       
def extract_feature_CV2(FileName,backbone= None,frames = 16, width=224, height=224):
    """get frames from video using cv2"""
    import cv2
    
    if backbone == None:ask_for_confirmation('No backbone provided....')
    backbone, preprocess_input = get_backbone(backbone)
    
    V = cv2.VideoCapture(FileName)
    duration = int(V.get(cv2.CAP_PROP_FRAME_COUNT))
    try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
    except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
    
    Frames = np.array([V.read()[1] for i in frame_id_list])
    Frames = np.array([cv2.resize(frame,(width,height)) for frame in Frames])
    features = np.array(backbone.predict(preprocess_input(Frames)))
    return features


def extract_features_DECORD(FileName,backbone, preprocessor_fn, frames = 16, width=224, height=224):
    try:
        V = VideoReader(FileName,  ctx=cpu(0), width= width, height=height)
        duration = len(V)
        try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
        except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
        
        Frames = V.get_batch(frame_id_list)
        del V
        gc.collect()
    except Exception as e:
        print(f"Decord decoding error{e}, using PIMS")
        return ExtractFeaturesPIMS(FileName, frames, width, height)
    return backbone.predict(preprocessor_fn(np.array(Frames)))

def ExtractFeaturesPIMS(FileName,backbone, preprocessor_fn,frames = 16, width=224, height=224):
    
    V = pims.Video(FileName)
    duration = len(V) 
    try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
    except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
    Frames = V[frame_id_list]
    #Frames = torch.tensor(Frames)
    del V 
    gc.collect()
    
    return backbone.predict(preprocessor_fn(np.array(Frames)))