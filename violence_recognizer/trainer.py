import sys

sys.path.append("..")

import numpy as np
import os, pprint
import config as cfg

from data_generator import TensorflowDataGenerator
import utility
from model import get_model, get_callbacks
from data_processing import Dataset, LabelingPattern
    
    

def define_model_configuration():
    model_configuration ={'Backbone':cfg.BACKBONE,
                        'MODEL NAME':cfg.PATH,
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


def prepare_dataset(file_pattern, debug= cfg.DEBUG):
    rlvs_dataset, my_dataset = None, None
    if cfg.USE_MY_DATASET:
        my_dataset   = Dataset(cfg.MY_VIDEO_FEATURES_PATH, cfg.MY_AUDIO_FEATURES_PATH,label_mapper= LabelingPattern.my_dataset,pattern = file_pattern)
        my_dataset.create_dataset()
        
    if cfg.USE_RLVS_DATASET:
        rlvs_dataset = Dataset(cfg.RLVS_VIDEO_FEATURES_PATH, cfg.RLVS_AUDIO_FEATURES_PATH,label_mapper= LabelingPattern.rlvs,pattern = file_pattern)
        rlvs_dataset.create_dataset()
        
    if cfg.CROSS_DATASET_VALIDATION in ['rlvs', 'my_dataset']:
        train_dataset,test_dataset  = (my_dataset,rlvs_dataset) if cfg.CROSS_DATASET_VALIDATION =='rlvs' \
                                        else (rlvs_dataset,my_dataset)   
    else:
        if my_dataset and rlvs_dataset:
            dataset = Dataset.merge_dataset([my_dataset,rlvs_dataset])
        else:
            dataset = my_dataset if my_dataset else rlvs_dataset
        train_dataset , test_dataset = dataset.make_test_split(cfg.TEST_PARTITION)
      
    return train_dataset , test_dataset 
   
def train_model(model,train_dataset ,save_history=True):
    
    train_data_generator = TensorflowDataGenerator(train_dataset, batch_size=cfg.BATCH_SIZE,val_partition=cfg.VAL_PARTITION)
    val_data = train_data_generator.load_val()
    history = model.fit(train_data_generator,
                        validation_data=val_data,
                        epochs=cfg.EPOCHS,
                        callbacks = get_callbacks(),
                        verbose=1)
    
    
    if save_history:
        np.save(f'{cfg.HISTORY_PATH}\history.npy',history.history)
        utility.dict_to_json(history.history,cfg.HISTORY_PATH+'\history.json')

    return model, history

def evaluate_model(model, test_dataset):
    model = get_model(dimension = cfg.MODEL_DIMENSION, summary=True)
    test_data_generator = TensorflowDataGenerator(test_dataset, batch_size=cfg.BATCH_SIZE,val_partition=0)
    
    prediction = model.evaluate_generator(test_data_generator,
                                          steps=len(test_data_generator),
                                          verbose=1)
    prediction = np.round(prediction,2)
    classification = np.rint(prediction).astype(int).flatten()
    
    return classification

def verbose_result(classification, test_labels, history,save_result=True):
    result = {  'Time':utility.times(),
                "SEED":cfg.SEED,
                'model_name':cfg.PATH,
                'My dataset':cfg.USE_MY_DATASET,
                'RLVS dataset':cfg.USE_RLVS_DATASET,
                'BACKBONE':cfg.BACKBONE,
                'INPUT_DIM':cfg.INPUT_DIM,
                "DISCARD SILENT VIDEO":cfg.DISCARD_SILENT_VIDEO,
                "Class Mapping":{'No Physical Violence':0,'Physical Violence': 1},
                "Statistics": utility.generate_report_skl(test_labels,classification),
                'Best model':utility.get_best_model_from_storage()
                }
    pprint.pprint(result) 
    if save_result: 
        utility.dict_to_json(result,cfg.RESULT_PATH+'\\result.json')
        utility.plot_loss(history)
        utility.plot_acc(history)
         
     
if __name__ == '__main__':
    train, test = prepare_dataset()
    
    





