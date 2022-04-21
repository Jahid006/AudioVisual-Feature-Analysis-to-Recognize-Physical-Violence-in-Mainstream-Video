import sys,os
sys.path.append('./violence_recognizer')

from violence_recognizer import trainer, model as modeling, predictor, data_generator
from violence_recognizer.data_processing import Dataset, LabelingPattern
import utility
import config as cfg

# next to do: adding argparse
#import argparse



file_pattern = {'video': ('@'+cfg.BACKBONE+'.'+str(16)+'.'+str('v0')+'.video.npy').lower()}
           #'audio': ('@'+'yamnet'+'.'+str(1)+'.'+str('v0')+'.audio.npy').lower()}
file_pattern['audio'] = '@yamnet.1.v0.audio.npy'


def evaluate_model(model, dataset,test_partition =.4,  save_result=False):
    test_datagen = data_generator.TensorflowDataGenerator(dataset, batch_size=cfg.BATCH_SIZE*2, val_partition=test_partition)
    test_data, test_label = test_datagen.load_val()
    
    #print([v.shape for k,v in test_data.items()], test_label.shape)
    
    classification = predictor.evaluate_model(model, test_data)
    print(classification.shape)
    #exit()
    predictor.verbose_result(classification, test_label, save_result=save_result)
    
    
def train(evaluate_trained_model=False):
    trainer.define_model_configuration()
    utility.make_dirs()
    model = modeling.get_model(dimension = cfg.MODEL_DIMENSION,
                           summary = True,
                           input_shape = cfg.INPUT_DIM)
    
    train_dataset, test_dataset = trainer.prepare_dataset(file_pattern)
    
    model, history = trainer.train_model(model,train_dataset,save_history=True)
    
    
    if evaluate_trained_model:
        evaluate_model(model, test_dataset)
        

def get_dataset(file_pattern):
    
    dataset_ = Dataset(video_features_path = cfg.RLVS_VIDEO_FEATURES_PATH,
                        audio_features_path = cfg.RLVS_AUDIO_FEATURES_PATH,
                        label_mapper = LabelingPattern.rlvs,
                        pattern = file_pattern)
    
    dataset_.create_dataset() 
    return dataset_
    

def get_model(model_path=None):
    model = modeling.get_model(dimension = cfg.MODEL_DIMENSION,
                           summary = True,
                           input_shape = cfg.INPUT_DIM)
    if model_path:
        model.load_weights(model_path)
    return model    

def main():
    TRAIN = False
    EVALUATE_MODEL = True
    if TRAIN:
        train()
    if EVALUATE_MODEL:
        model = get_model(r"F:\S-Home\ViolenceRecognizer\data\saved_models\resnet50_my_dataset_rlvs_dataset_v0\model_015_0.009_1.000_.h5")
        dataset_ = get_dataset(file_pattern)
        evaluate_model(model, dataset_, test_partition=1)
        
        
if __name__ == '__main__':
    main()