BACKBONES = ['Resnet50', 'DenseNet201','InceptionV3', 'I3D','P3D','R2+1D']
INPUT_DIMS ={'Resnet50':(16,2048),
             'DenseNet201':(16,1920),
             'InceptionV3':(16,2048),
             'I3D':(1,2048),
             'P3D':(2,2048),
             'R2+1D':(1,2048) 
             }

SEGMENTS ={  'Resnet50':16,
             'DenseNet201':16,
             'InceptionV3':16,
             'I3D':10,
             'P3D':5,
             'R2+1D': 10
             }



######################################## Changable Configuration #######################################
BACKBONE = BACKBONES[0]
USE_MY_DATASET = True
USE_RLVS_DATASET = True

CROSS_DATASET_VALIDATION = '' #could be 'rlvs' or 'my_dataset'
ONLY = '_only' if ((USE_MY_DATASET ^ USE_RLVS_DATASET) and CROSS_DATASET_VALIDATION == '') else ''
TEST_PARTITION = .1
VAL_PARTITION = .1

CHECK_CONFIRMATION = True
DISCARD_SILENT_VIDEO = False
SEED = 0
DEBUG = True
EPOCHS = 50
LEARNING_RATE = .003
BATCH_SIZE =32
MODEL_DIMENSION = 256
VERSION = 'v0'
NUM_SEGMENTS = SEGMENTS[BACKBONE]



###############################################--PATH--##################################################
RLVS_VIDEO_FEATURES_PATH = r"F:\S-Home\ViolenceRecognizer\data\dataset\rlvs"
RLVS_AUDIO_FEATURES_PATH = r"F:\S-Home\ViolenceRecognizer\data\dataset\rlvs"
MY_VIDEO_FEATURES_PATH = r"F:\S-Home\ViolenceRecognizer\data\dataset\my_dataset"
MY_AUDIO_FEATURES_PATH = r"F:\S-Home\ViolenceRecognizer\data\dataset\my_dataset"


############################################# Prediction Config #############################################
END_TO_END_TESTING = False
TEST_VIDEO_PATH = './data/test_data' #mp4 only
TRAINED_MODEL_PATH = ''




######################################---Auto-Config---###############################################
INPUT_DIM = INPUT_DIMS[BACKBONE]

MXNET_MODEL = ['I3D','P3D','R2+1D']
IF_MXNET_MODEL = BACKBONE in MXNET_MODEL

PATH = ''.join([BACKBONE,
                '_my_dataset' if USE_MY_DATASET else '',
                '_rlvs_dataset' if USE_RLVS_DATASET else '',
                '_only' if ((USE_MY_DATASET ^ USE_RLVS_DATASET) and CROSS_DATASET_VALIDATION == '') else '',
                '_discard_silent_video' if DISCARD_SILENT_VIDEO else '',
                f'_{VERSION}' if VERSION else '']).lower()
        
HISTORY_PATH = './data/models_history/' + PATH
LOG_PATH = './data/tensorboard_log/' + PATH
MODEL_PATH = './data/saved_models/' + PATH
RESULT_PATH = './data/generated_result/' + PATH
