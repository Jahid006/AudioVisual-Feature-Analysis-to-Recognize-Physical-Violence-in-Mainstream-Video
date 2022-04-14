import glob, os, logging

def get_files(video_files_path, extensions = ['.mp4','.avi','.mov']):
    
    files = []
    for extension in extensions:
        files.extend(glob.glob(video_files_path+'/**/*'+extension,recursive=True))
    return files
    
def make_text_file(video_files_path, text_file_path = 'video.txt'):
    
    files = get_files(video_files_path) 
    
    # if os.path.exists(text_file_path):
    #     return text_file_path
    with open(text_file_path,'w') as f:
        for file in files:
            f.write(file+ '  1  1 '+'\n')
    return text_file_path

def get_save_dir(z, model_name,num_segment, version, to_replace = '/video'):
    feat_file = z.replace(z[-4:], '.npy')
    feat_file = feat_file.replace('/dataset/', '/features/')
    feat_file = feat_file.replace(to_replace, f"{to_replace}/{model_name.lower().split('_')[0]+'.'+str(num_segment)}s{version}")    
    return feat_file

def set_logger():
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    
    return logger

class CONFIG_MXNET:
    version = ''
    data_list= make_text_file(video_files_path= "F:\S-Home\ViolenceRecognizer\data\\test_data", #'F:/S-Home/r2/data/dataset/my_dataset/video',
                              text_file_path ='video.txt')
    version = ''
    save_dir='./features'
    dtype='float32'
    model='i3d_resnet50_v1_kinetics400'
    input_size=224
    num_segments=10
    new_height=256
    new_length=32
    new_step=1
    new_width=340
    num_classes=400
    
    data_aug='v1'
    data_dir=''
    fast_temporal_stride=2
    gpu_id=0
    hashtag=''
    log_interval=10
    mode=None
    need_root=False
    num_crop=1
    resume_params=''
    slow_temporal_stride=16
    slowfast=False
    ten_crop=False
    three_crop=False
    use_decord=True
    use_pretrained=True
    video_loader=True
    logger = set_logger()
    get_save_dir = get_save_dir
    
class CONFIG:
    BACKBONES = ['Resnet50', 'DenseNet201','InceptionV3']
    backbone_name = BACKBONES[1]
    height = 224
    width = 224
    frames = 16
    video_files_path= "F:\S-Home\ViolenceRecognizer\data\\test_data"
    #'F:/S-Home/r2/data/dataset/my_dataset/video'
    video_file_list = get_files(video_files_path)
    get_save_dir = get_save_dir
    version = ''
    
class AUDIO_CONFIG:
    model_name ='yamnet'
    frames = 16000
    is_mono = True
    num_segments = 1
    video_files_path= "F:\S-Home\ViolenceRecognizer\data\\test_data"
    video_file_list = get_files(video_files_path)
    get_save_dir = get_save_dir
    version = ''
    to_replace = '/audio'
    