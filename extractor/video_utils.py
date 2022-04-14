import pims, numpy as np, gc
from decord import VideoReader,  cpu
import decord

def ExtractFeatureDECORD(FileName,frames = 16, width=224, height=224):
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
        return ExtractFeaturePIMS(FileName, frames, width, height)
    return Frames

def ExtractFeaturePIMS(FileName,frames = 16, width=224, height=224):
    
    V = pims.Video(FileName)
    duration = len(V) 
    try:    frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=False))
    except: frame_id_list = np.sort(np.random.choice(range(duration), frames, replace=True))
    Frames = V[frame_id_list]
    #Frames = torch.tensor(Frames)
    del V 
    gc.collect()
    
    return np.array(Frames)