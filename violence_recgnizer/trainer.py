import numpy as np
import os, pprint

from config import BACKBONE, USE_MY_DATASET, USE_RLVS_DATASET, INPUT_DIM, IF_MXNET_MODEL,\
    PATH, HISTORY_PATH, LOG_PATH, MODEL_PATH, RESULT_PATH, DEBUG, CROSS_DATASET, PARTITION,\
    DISCARD_SILENT_VIDEO, SEED, EPOCHS, LEARNING_RATE
    
from utility import times, dict_to_json, generate_report_skl, ask_for_confirmation, plot_acc,plot_loss
from model import get_model, get_callbacks, get_best_model_from_storage
from data_processing import get_my_data, get_rlvs_data,get_all_data, get_shuffled_data, split_train_test_data
    
    

def define_model_configuration():
    model_configuration ={'Backbone':BACKBONE,
                        'MODEL NAME':PATH,
                        "SEED": SEED,
                        "LEARNING_RATE": LEARNING_RATE,
                        "EPOCHS": EPOCHS,
                        'Use_My_Dataset':USE_MY_DATASET,
                        'Use_Rlvs_Dataset':USE_RLVS_DATASET,
                        'Input_Dim':INPUT_DIM,
                        'If_Mxnet_Model':IF_MXNET_MODEL,
                        'History_Path':HISTORY_PATH,
                        'Log_Path':LOG_PATH,
                        'Model_Path':MODEL_PATH,
                        'Result_Path':RESULT_PATH,
                        'Debug':DEBUG,
                        'Cross_Dataset':CROSS_DATASET,
                        'Partition':PARTITION}
    pprint.pprint(model_configuration)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    else:
        ask_for_confirmation('Directories already exists. Proceeding to train a model with\
            this configuration might conflict with existing trained model.')  
        
def make_dirs():
    _ = [make_dir(i) for i in [HISTORY_PATH, LOG_PATH, MODEL_PATH, RESULT_PATH]]

def dataset(debug= DEBUG):
    videos, audios, labels = get_all_data(my_data = USE_MY_DATASET, rlvs = USE_RLVS_DATASET, DEBUG=debug)
    videos, audios, labels, perm, Iperm = get_shuffled_data(videos, audios, labels)
    train_videos, train_audios, train_labels, test_videos, test_audios, test_labels = split_train_test_data(videos, audios, labels)
    if len(CROSS_DATASET):
        print('\n','Cross dataset evaluation: ',CROSS_DATASET)
        test_videos, test_audios, test_labels = get_rlvs_data() if CROSS_DATASET =='rlvs' else get_my_data()
        
    print(f'\n{len(train_videos)} train videos, {len(test_videos)} test videos')
    
    return train_videos, train_audios, train_labels, test_videos, test_audios, test_labels, perm, Iperm
   
def train_model(train_videos, train_audios, train_labels,save_history=True):
    
    model = get_model(dimension = 256, summary=True)
    history = model.fit([train_videos,train_audios],train_labels, epochs=EPOCHS,
                    verbose=1,steps_per_epoch=32,validation_split=.1,shuffle = True,
                    callbacks = get_callbacks())
    
    if save_history:
        np.save(f'{HISTORY_PATH}\history.npy',history.history)
        dict_to_json(history.history,HISTORY_PATH+'\history.json')

    return model, history

def evaluate_model(model, test_videos, test_audios):
    prediction = model.predict([test_videos,test_audios],verbose=1,steps=len(test_videos)//64)
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    
    return classification

def verbose_result(classification, test_labels, history,save_result=True):
    result = {  'Time':times(),
                "SEED":SEED,
                'model_name':PATH,
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
        dict_to_json(result,RESULT_PATH+'\\result.json')
        plot_loss(history)
        plot_acc(history)
         
def main():
    define_model_configuration()
    ask_for_confirmation('Directories will be created according to the configuration.')
    make_dirs()
    
    train_videos, train_audios, train_labels, test_videos, test_audios, test_labels, perm, Iperm = dataset(False)
    
    ask_for_confirmation("let's  train the model...")
    model, history = train_model(train_videos, train_audios, train_labels,save_history=True)
    
    ask_for_confirmation("Evaluate the model...")
    classification = evaluate_model(model, test_videos, test_audios)
    
    ask_for_confirmation("let's generate verbose result...")
    verbose_result(classification, test_labels,history=history)
        
if __name__ == '__main__':
    main()
    
    





