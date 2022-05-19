import numpy as np
import cv2, gc,os,glob,tqdm
import pims
import decord
from decord import VideoReader, gpu, cpu
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def show_sampled_image(images,grid_width = 4 ,title=None):
    import matplotlib.pyplot as plt
    _, axs = plt.subplots(grid_width, grid_width, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img)
    plt.show()

def ExtractFeatureUsingPIMS(FileName,Model= None,preprocess_input= None,frames = 16, width=224, height=224):
    #from tensorflow.keras.applications.densenet import preprocess_input
    V = pims.Video(FileName) 
    duration = len(V) 
      
    try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
    except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
    
    Frames = np.array(V[frame_id_list])
    Frames = np.array([cv2.resize(frame,(width,height)) for frame in Frames])
    features = np.array(Model.predict(preprocess_input(Frames)))
    return features, Frames

def ExtractFeatureUsingDECORD(FileName,Model= None,preprocess_input= None,frames = 16, width=224, height=224):
        #from tensorflow.keras.applications.densenet import preprocess_input
        try:
            V = VideoReader(FileName,  ctx=cpu(0), width= width, height=height)
            duration = len(V)
            try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
            except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
            
            Frames = V.get_batch(frame_id_list).asnumpy()
            features = np.array(Model.predict(preprocess_input(Frames)))
            del V
            gc.collect()
        except Exception as e:
            print(f"Decord decoding error{e}, using PIMS")
            return ExtractFeatureUsingPIMS(FileName,Model,preprocess_input,frames, width, height)
        return features, Frames

def ExtractFeatureUsingCV2(FileName,Model= None,preprocess_input= None,frames = 16, width=224, height=224):
    """get frames from video using cv2"""
    V = cv2.VideoCapture(FileName)
    duration = int(V.get(cv2.CAP_PROP_FRAME_COUNT))
    try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
    except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
    
    sampling_rate = int(V.get(cv2.CAP_PROP_FPS))*duration//frames
    
    frame_counter = 0
    Frames = []
    while V.isOpened():
        r, frame = V.read()
        if r == False:break
        if frame_counter % sampling_rate == 0:
            Frames.append(cv2.resize(frame,(width,height)))
        frame_counter+=1
        
    if len(Frames)<frames: Frames.extend([Frames[-1]*(frames-len(Frames))])
    
    Frames = np.array(Frames)
    return np.array(Model.predict(preprocess_input(Frames))), Frames
    


def main(show_sampled_image_=False):

    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input
    model = tf.keras.applications.DenseNet201(include_top=False,weights="imagenet",pooling='avg')

    # data  = glob.glob(r'F:\S-Home\RecognizingPhysicalViolence\dataset\my_dataset\video\**\*.mp4',recursive=True)
    # data.extend(glob.glob(r'F:\S-Home\RecognizingPhysicalViolence\dataset\rlvs\video\**\*.avi',recursive=True))
    # data.extend(glob.glob(r'F:\S-Home\RecognizingPhysicalViolence\dataset\rlvs\video\**\*.mp4',recursive=True))
    # data = np.array(data)
    
    data = glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\movies\\**\\*.avi',recursive=True)
    data.extend(glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\Peliculas\\**\\*.avi',recursive=True))
    data.extend(glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\Peliculas\\**\\*.mpg',recursive=True))
    
    print(len(data))#,data[0], data[-1])
    
    exit()

    #out_dirs = r"F:\S-Home\RecognizingPhysicalViolence\video_features\dataset_split\DenseNet201"
    
    if show_sampled_image_:
        choosen = np.random.choice(len(data),1) 
        data = data[choosen]
        
    for video in tqdm.tqdm(data[:]):
        name = video.split('\\')[-1][:-4]
        dataset_split = video.split('\\')[-4]
        dataset_class = video.split('\\')[-2]
        
        
        
        name =  f"{out_dirs.replace('dataset_split',dataset_split)}\{dataset_class}\{name}.npy"
        os.makedirs(out_dirs.replace('dataset_split',dataset_split)+'\\'+dataset_class,exist_ok=True)

        features,Frames = ExtractFeatureUsingDECORD(video, model, preprocess_input)
        
        if show_sampled_image_:
            show_sampled_image(Frames,title=name)
            return 
        np.save(name,features,allow_pickle=True)
        
def extract():

    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input
    model = tf.keras.applications.DenseNet201(include_top=False,weights="imagenet",pooling='avg')

    data = glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\movies\\**\\*.avi',recursive=True)
    data.extend(glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\Peliculas\\**\\*.avi',recursive=True))
    data.extend(glob.glob(r'F:\S-Home\\RecognizingPhysicalViolence\dataset\\Violence_dataset_in_Research\\Peliculas\\**\\*.mpg',recursive=True))
    
    for video in tqdm.tqdm(data[:]): 
        name =  video[:-4]+'.npy'
        print(name)
        features,_ = ExtractFeatureUsingDECORD(video, model, preprocess_input)
        np.save(name,features,allow_pickle=True)
        
        
if __name__ == "__main__":
    extract()