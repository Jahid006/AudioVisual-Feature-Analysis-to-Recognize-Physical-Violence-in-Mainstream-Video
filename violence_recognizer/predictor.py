import numpy as np
import pprint, glob
import config as cfg

import sys
sys.path.append('..')
    
import utility 
from model import get_model, get_best_model_from_storage

    
def LoadModel(model_path= None):
    define_model_configuration()
    utility.ask_for_confirmation("let's load the model...")
    
    model_path = model_path if model_path else cfg.MODEL_PATH
    model = get_model(cfg.MODEL_DIMENSION, summary=False)
    model.load_weights(model_path)

    utility.ask_for_confirmation(f'No model found named {model_path}. Please train a model with the configuration stated above.')
    return model


def define_model_configuration():
    model_configuration ={'Backbone':cfg.BACKBONE,
                        'MODEL NAME':cfg.TRAINED_MODEL_PATH,
                        "SEED": cfg.SEED,
                        "LEARNING_RATE": cfg.LEARNING_RATE,
                        "EPOCHS": cfg.EPOCHS,
                        'Use_My_Dataset':cfg.USE_MY_DATASET,
                        'Use_Rlvs_Dataset':cfg.USE_RLVS_DATASET,
                        'Input_Dim':cfg.INPUT_DIM,
                        'If_Mxnet_Model':cfg.IF_MXNET_MODEL,
                        'History_Path':cfg.HISTORY_PATH,
                        'Log_Path':cfg.LOG_PATH,
                        'Model_Path':cfg.MODEL_PATH,
                        'Result_Path':cfg.RESULT_PATH,
                        'Debug':cfg.DEBUG,
                        'Cross_Dataset':cfg.CROSS_DATASET_VALIDATION,
                        'Val_Partition':cfg.VAL_PARTITION,
                        'Test_Partition':cfg.TEST_PARTITION}
    pprint.pprint(model_configuration)


def evaluate_model(model, test_data):
    prediction = model.predict(test_data,verbose=1,batch_size = cfg.BATCH_SIZE*2)
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    return classification


def verbose_result(classification, test_labels, history=None,save_result=True):
    result = {'Time':utility.times(),
          'model_name':cfg.PATH,
          "SEED":cfg.SEED,
          'My dataset':cfg.USE_MY_DATASET,
          'RLVS dataset':cfg.USE_RLVS_DATASET,
          'BACKBONE':cfg.BACKBONE,
          'INPUT_DIM':cfg.INPUT_DIM,
          "DISCARD SILENT VIDEO":cfg.DISCARD_SILENT_VIDEO,
          "Class Mapping":{'No Physical Violence':0,'Physical Violence': 1},
          "Statistics": utility.generate_report_skl(test_labels.reshape(-1),classification.reshape(-1)),
          'Best model':get_best_model_from_storage()
          }
    pprint.pprint(result) 
    if save_result: 
        utility.dict_to_json(result,cfg.RESULT_PATH+'\\result_testing.json')

    
def test():
    model = LoadModel()
    utility.ask_for_confirmation('Lets Load the data...')
    
    if cfg.IF_MXNET_MODEL:
        print('END_TO_END_TESTING is not available for MXNET models.')
        return 
    files = glob.glob(cfg.TEST_VIDEO_PATH+'\\*.mpg')
    files.extend(glob.glob(cfg.TEST_VIDEO_PATH+'\\*.avi'))
    files.extend(glob.glob(cfg.TEST_VIDEO_PATH+'\\*.mp4'))
    
    utility.ask_for_confirmation(f'Total videos: {len(files)}')
    
    Audio_Embedding = []
    Video_Embedding = []
    
    if backbone == None:utility.ask_for_confirmation('No backbone provided....')
    backbone, preprocess_input = utility.get_backbone(backbone)
    
    for f in files:
        utility.video_to_wav(f)
        Audio_Embedding.append(utility.generate_audio_embedding(f[:-4]+'.wav'))
        Video_Embedding.append(utility.extract_features_DECORD(f,backbone, preprocess_input))
        
    Audio_Embedding = np.array(Audio_Embedding)
    Video_Embedding = np.array(Video_Embedding)
    
    model_input = {'audio': Audio_Embedding,
                   'video': Video_Embedding}
    
    prediction = model.predict(model_input)
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    
    print(list(zip(files,classification,prediction)))    
    
if __name__ == '__main__':
    
    if cfg.END_TO_END_TESTING:
        test()
        
    






