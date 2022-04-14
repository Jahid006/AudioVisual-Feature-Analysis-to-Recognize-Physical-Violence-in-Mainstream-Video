BACKBONES = ['Resnet50', 'DenseNet201','InceptionV3',
             'I3D','P3D','R2+1D']

######################################## Changable Configuration #######################################
BACKBONE = BACKBONES[1]
USE_MY_DATASET = False
USE_RLVS_DATASET = True

CROSS_DATASET = '' #could be 'rlvs' or 'my_dataset'
ONLY = '_only' if ((USE_MY_DATASET ^ USE_RLVS_DATASET) and CROSS_DATASET == '') else ''
PARTITION = 1 if CROSS_DATASET else .9

CHECK_CONFIRMATION = True
DISCARD_SILENT_VIDEO = False
SEED = 0
DEBUG = True
EPOCHS = 50
LEARNING_RATE = .001
ID = 'V2'

############################################# Prediction Config #############################################
END_TO_END_TESTING = False
TEST_VIDEO_PATH = 'test_data' #mp4 only
########################################################################################################


INPUT_DIMS ={'Resnet50':(16,2048),
             'DenseNet201':(16,1920),
             'InceptionV3':(16,2048),
             'I3D':(1,2048),
             'P3D':(2,2048),
             'R2+1D':(1,2048) 
             }

INPUT_DIM = INPUT_DIMS[BACKBONE]

MXNET_MODEL = ['I3D','P3D','R2+1D']
IF_MXNET_MODEL = BACKBONE in MXNET_MODEL

PATH = ''.join([BACKBONE,
                '_my_dataset' if USE_MY_DATASET else '',
                '_rlvs_dataset' if USE_RLVS_DATASET else '',
                '_only' if ((USE_MY_DATASET ^ USE_RLVS_DATASET) and CROSS_DATASET == '') else '',
                '_discard_silent_video' if DISCARD_SILENT_VIDEO else '',
                f'_{ID}' if ID else '']) 
        
HISTORY_PATH = './models_history/' + PATH
LOG_PATH = './tensorboard_log/' + PATH
MODEL_PATH = './saved_models/' + PATH
RESULT_PATH = './generated_result/' + PATH
