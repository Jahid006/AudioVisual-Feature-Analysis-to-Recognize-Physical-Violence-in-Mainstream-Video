import tensorflow as tf
import random, numpy as np
from config import INPUT_DIM

from data_processing import Dataset, LabelingPattern

class TensorflowDataGenerator(tf.keras.utils.Sequence): #
    def __init__(self, dataset, batch_size=32, val_partition=0.2):
        X = dataset.zipped_features_path 
        Y = dataset.labels
        self.shuffle = True
        self.XY = list(zip(X, Y))
        random.seed(0)
        random.shuffle(self.XY)  # shuffle it randomly
        
        self.train_XY = self.XY[:int(1-val_partition*len(self.XY))]
        self.val_XY = self.XY[-int(val_partition*len(self.XY)):]
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.train_XY)
    
    def __len__(self):
        return len(self.train_XY) // self.batch_size
    
    def __getitem__(self, index):
        batch = self.train_XY[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__get_data(batch)
        return X, Y

    def __get_data(self, batch):
        X_ = {'video': [], 'audio': []}
        Y_=[]
        for x_,y_ in batch:
            video_feature  = np.load(x_[0],allow_pickle=True).reshape(INPUT_DIM)
            audio_feature  = np.load(x_[1],allow_pickle=True).flatten() if x_[1] != '[silent_video]' else np.zeros(1024).flatten()
            
            X_['video'].append(video_feature)
            X_['audio'].append(audio_feature)
            
            Y_.append(y_)
            
        X_['video'] = np.array(X_['video'])
        X_['audio'] = np.array(X_['audio'])
        
        return X_,  np.array(Y_).reshape((-1,1))
    
    def load_val(self):
        X_ = {'video': [], 'audio': []}
        Y_=[]
        for x_,y_ in self.val_XY:
            video_feature  = np.load(x_[0],allow_pickle=True).reshape(INPUT_DIM)
            audio_feature  = np.load(x_[1],allow_pickle=True).flatten() if x_[1] != '[silent_video]' else np.zeros(1024).flatten()
            
            X_['video'].append(video_feature)
            X_['audio'].append(audio_feature)
            Y_.append(y_)
            
        X_['video'] = np.array(X_['video'])
        X_['audio'] = np.array(X_['audio'])
        
        return X_,  np.array(Y_).reshape((-1))
    
    
if __name__=="__main__":
    rlvs = Dataset(video_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\rlvs\video\i3d.10s",
                   audio_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\rlvs\audio\yamnet",
                   label_mapper = LabelingPattern.rlvs)
    
    my_dataset = Dataset(video_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\my_dataset\video\i3d.10s",
                    audio_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\my_dataset\audio\yamnet",
                    label_mapper = LabelingPattern.my_dataset)

    rlvs.create_dataset()
    my_dataset.create_dataset()
    combined = Dataset.merge_dataset([rlvs, my_dataset])
    
    train_data_generator = TensorflowDataGenerator(combined, batch_size=32,val_partition=.1)
    val_data = train_data_generator.load_val()
    val_data[0]['video'].shape