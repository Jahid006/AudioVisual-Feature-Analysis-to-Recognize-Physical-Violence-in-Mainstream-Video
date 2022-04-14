from IPython.lib.display import Audio
import numpy as np
import pprint, glob

from config import BACKBONE, END_TO_END_TESTING, USE_MY_DATASET, USE_RLVS_DATASET, INPUT_DIM, IF_MXNET_MODEL,SEED,\
    PATH, HISTORY_PATH, LOG_PATH, MODEL_PATH, RESULT_PATH, DEBUG, CROSS_DATASET, PARTITION,DISCARD_SILENT_VIDEO,\
    END_TO_END_TESTING, TEST_VIDEO_PATH
    
from utility import times, dict_to_json, generate_report_skl, Exit, video_to_wav, generate_audio_embedding,extract_feature_CV2
from model import get_model, get_best_model_from_storage
from data_processing import get_my_data, get_rlvs_data,get_all_data, get_shuffled_data, split_train_test_data
    
    

def LoadModel():
    define_model_configuration()
    Exit("let's load the model...")
    model = get_model(dimension = 256, summary=False)
    if get_best_model_from_storage():
        model.load_weights(get_best_model_from_storage())
        return model
    Exit(f'No model found named {PATH}. Please train a model with the configuration stated above.')
    return model

def define_model_configuration():
    model_configuration ={'Backbone':BACKBONE,
                        'Use_My_Dataset':USE_MY_DATASET,
                        'Use_Rlvs_Dataset':USE_RLVS_DATASET,
                        'Input_Dim':INPUT_DIM,
                        'If_Mxnet_Model':IF_MXNET_MODEL,
                        'Path':PATH,
                        "SEED":SEED,
                        'History_Path':HISTORY_PATH,
                        'Log_Path':LOG_PATH,
                        'Model_Path':MODEL_PATH,
                        'Result_Path':RESULT_PATH,
                        'Debug':DEBUG,
                        'Cross_Dataset':CROSS_DATASET,
                        'Partition':PARTITION}
    pprint.pprint(model_configuration)

def evaluate_model(model, test_videos, test_audios):
    prediction = model.predict([test_videos,test_audios],verbose=1,steps=len(test_videos)//64)
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    
    return classification

def verbose_result(classification, test_labels, history=None,save_result=True):
    result = {'Time':times(),
          'model_name':PATH,
          "SEED":SEED,
          'My dataset':USE_MY_DATASET,
          'RLVS dataset':USE_RLVS_DATASET,
          'BACKBONE':BACKBONE,
          'INPUT_DIM':INPUT_DIM,
          "DISCARD SILENT VIDEO":DISCARD_SILENT_VIDEO,
          "Class Mapping":{'No Physical Violence':0,'Physical Violence': 1},
          "Statistics": generate_report_skl(test_labels,classification),
          'Best model':get_best_model_from_storage()
          }
    pprint.pprint(result) 
    if save_result: 
        dict_to_json(result,RESULT_PATH+'\\result_testing.json')
        
def dataset(debug= DEBUG):
    videos, audios, labels = get_all_data(my_data = USE_MY_DATASET, rlvs = USE_RLVS_DATASET, DEBUG=debug)
    videos, audios, labels, perm, Iperm = get_shuffled_data(videos, audios, labels)
    train_videos, train_audios, train_labels, test_videos, test_audios, test_labels = split_train_test_data(videos, audios, labels)
    if len(CROSS_DATASET):
        print('\n','Cross dataset evaluation: ',CROSS_DATASET)
        test_videos, test_audios, test_labels = get_rlvs_data() if CROSS_DATASET =='rlvs' else get_my_data()
        
    print(f'\n{len(train_videos)} train videos, {len(test_videos)} test videos')
    
    return train_videos, train_audios, train_labels, test_videos, test_audios, test_labels, perm, Iperm  
    
def main(): 
    model = LoadModel()
    Exit('Lets Load the dataset...')
    
    train_videos, train_audios, train_labels, test_videos, test_audios, test_labels, perm, Iperm = dataset()
    
    Exit("Evaluate the model...")
    classification = evaluate_model(model, test_videos, test_audios)
    
    Exit("let's generate verbose result...")
    verbose_result(classification, test_labels, save_result=False)
    
def test():
    model = LoadModel()
    Exit('Lets Load the data...')
    
    if IF_MXNET_MODEL:
        print('END_TO_END_TESTING is not available for MXNET models.')
        return 
    files = glob.glob(TEST_VIDEO_PATH+'\\*.mpg')
    files.extend(glob.glob(TEST_VIDEO_PATH+'\\*.avi'))
    files.extend(glob.glob(TEST_VIDEO_PATH+'\\*.mp4'))
    
    Exit(f'Total videos: {len(files)}')
    
    Audio_Embedding = []
    Video_Embedding = []
    
    
    
    for f in files:
        video_to_wav(f)
        Audio_Embedding.append(generate_audio_embedding(f[:-4]+'.wav'))
        Video_Embedding.append(extract_feature_CV2(f,BACKBONE))
        
    Audio_Embedding = np.array(Audio_Embedding)
    Video_Embedding = np.array(Video_Embedding)
    
    prediction = model.predict([Video_Embedding,Audio_Embedding])
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    
    print(list(zip(files,classification,prediction)))    
    
if __name__ == '__main__':
    
    if not END_TO_END_TESTING:
        main()
    else:
        test()
        
    






