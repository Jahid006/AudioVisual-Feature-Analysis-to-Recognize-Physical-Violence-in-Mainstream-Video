import numpy as np
import glob,os

from utility import shuffle_index, ask_for_confirmation
from config import BACKBONE, DISCARD_SILENT_VIDEO, PARTITION



def get_my_data(DEBUG=False):
    positive_video = glob.glob(f'video_features\my_dataset\{BACKBONE}\\notPhysicalViolence\*')
    negative_video = glob.glob(f'video_features\my_dataset\{BACKBONE}\PhysicalViolence\*')
    
    positive_audio_embedding = glob.glob(r'audio_features\my_dataset\notPhysicalViolence\*')
    negative_audio_embedding = glob.glob(f'audio_features\my_dataset\PhysicalViolence\*')
    
    video_files = positive_video + negative_video
    audio_files = positive_audio_embedding + negative_audio_embedding
    
    print(f'{len(video_files)} videos and {len(audio_files)} audio files found in my dataset')
    
    videos = [np.load(video_files[k],allow_pickle=True) for k in range(len(video_files))]
    audios = [np.load(audio_files[k],allow_pickle=True).flatten() for k in range(len(audio_files))]
    labels = [0]*len(positive_video) + [1]*len(negative_video)
    
    videos = np.array(videos,dtype=np.float32)
    audios = np.array(audios,dtype=np.float32)
    labels = np.array(labels,dtype=np.int32)
    
    if DEBUG:
        idx = np.random.choice(range(len(videos)),size=10,replace=False)
        print('\n',list(zip(np.array(video_files)[idx],np.array(audio_files)[idx],labels[idx])))

    print('my data shape: ',videos.shape,audios.shape,labels.shape)
    
    return videos, audios, labels


rlvs_audios_embedding_not_found = []

def get_audio_embedding(audio_file):
    if os.path.exists(audio_file):
        return np.load(audio_file,allow_pickle=True).flatten()
    
    rlvs_audios_embedding_not_found.append(audio_file)
    return np.zeros(1024).flatten()

def naming(video_file):
    name  = video_file.split('\\')[-1][:-4]
    if name.split('.')[-1] in ['i3d10s','p3d5s','r21d']:
        name = name.replace('.'+name.split('.')[-1],'')
    return f'audio_features\\rlvs\{name}.npy'

def get_rlvs_data(DEBUG=False): 
    video_files = np.array(glob.glob(f'video_features\\rlvs\{BACKBONE}\*'))
    
    print(f'{len(video_files)} videos found in rlvs dataset')
    
    audio_available = np.array([os.path.exists(naming(video_file)) for video_file in video_files])
    
    if DISCARD_SILENT_VIDEO: 
        ask_for_confirmation(f'{len(video_files) - np.sum(audio_available)} videos are discarded, because they are silent')
        video_files = video_files[audio_available]

    videos = [np.load(video_files[k],allow_pickle=True) for k in range(len(video_files))]
    audios = [get_audio_embedding(naming(video_files[k])) for k in range(len(video_files))]
    
    class_map = {'NV':0,'V':1}
    labels = [class_map[i.split('\\')[-1].split('_')[0]] for i in video_files]
    
    videos = np.array(videos,dtype=np.float32)
    audios = np.array(audios,dtype=np.float32)
    labels = np.array(labels,dtype=np.int32)
    
    if DEBUG:
        idx = np.random.choice(range(len(videos)),size=10,replace=False)
        print('\n',list(zip(np.array(video_files)[idx],
                       [(naming(k)) for k in np.array(video_files)[idx]],
                       labels[idx])))
    
    print("rlvs data shape: ",videos.shape,audios.shape,labels.shape)
    print(f'{len(rlvs_audios_embedding_not_found)} audio files not found')
    
    return videos, audios, labels

def get_all_data(my_data = False, rlvs = False,DEBUG=False):
    videos, audios, labels = None, None, None
    
    if my_data:
        my_videos, my_audios, my_labels = get_my_data(DEBUG)
        videos = np.concatenate((videos,my_videos), axis = 0) if videos is not None else my_videos
        audios = np.concatenate((audios,my_audios), axis = 0) if audios is not None else my_audios
        labels = np.concatenate((labels,my_labels), axis = 0) if labels is not None else my_labels

    if rlvs:
        rlvs_videos, rlvs_audios, rlvs_labels = get_rlvs_data(DEBUG)
        videos = np.concatenate((videos,rlvs_videos), axis = 0) if videos is not None else rlvs_videos
        audios = np.concatenate((audios,rlvs_audios), axis = 0) if audios is not None else rlvs_audios
        labels = np.concatenate((labels,rlvs_labels), axis = 0) if labels is not None else rlvs_labels
     
    return videos, audios, labels

def get_shuffled_data(videos, audios, labels,from_storage= None):
    
    if from_storage: 
        permutation, inversePermutation = np.load('history\permutation_index.npy',allow_pickle=True)
    else:  permutation, inversePermutation = shuffle_index(len(videos))
    
    videos = videos[permutation]
    audios = audios[permutation]
    labels = labels[permutation]
    
    #np.save('config\permutation_index.npy',[permutation, inversePermutation],allow_pickle=True)
    return videos, audios, labels, permutation, inversePermutation   

def split_train_test_data(videos, audios, labels,partition = PARTITION):
    """split data into train, test, validation"""
    
    if partition<=1:  partition = int(len(videos)*partition)
    
    train_videos = videos[:partition]
    train_audios = audios[:partition]
    train_labels = labels[:partition]
    
    test_videos = videos[partition:]
    test_audios = audios[partition:]
    test_labels = labels[partition:]

    
    
    return train_videos, train_audios, train_labels, test_videos, test_audios, test_labels
    
def get_data_generator(videos, audios, labels, batch_size):
    while True:
        permutation, inversePermutation = shuffle_index(len(videos))
        videos = videos[permutation]
        audios = audios[permutation]
        labels = labels[permutation]
        
        for i in range(0,len(videos),batch_size):
            yield videos[i:i+batch_size], audios[i:i+batch_size], labels[i:i+batch_size]


