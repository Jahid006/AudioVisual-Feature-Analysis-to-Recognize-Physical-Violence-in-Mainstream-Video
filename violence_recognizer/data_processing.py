import re
import copy
from itertools import compress
from sklearn.utils import shuffle
import glob,sys
sys.path.append('..')

import config as cfg

class LabelingPattern:
    def rlvs(video_file_path):
        return  0 if 'NonViolence' in video_file_path else 1
    def my_dataset(video_file_path):
        return  0 if 'notPhysicalViolence' in video_file_path else 1


class Dataset:
    def __init__(self,video_features_path, audio_features_path, label_mapper, pattern) -> None:
        self.video_features_path = glob.glob(f'{video_features_path}\\**\\*{pattern["video"]}', recursive=True)
        self.audio_features_path = glob.glob(f'{audio_features_path}\\**\\*{pattern["audio"]}', recursive=True)
        self.silent_video_list = []
        self.label_mapper = label_mapper
        self.zipped_features_path = []
        self.labels  = []
         
    def create_dataset(self):
        self.get_zipped_path()
        self.labels = self.map_label()
        
        print(f'{len(self.video_features_path)} videos and {len(self.audio_features_path)} audio files found in this dataset')
        assert len(self.zipped_features_path) == len(self.labels), f"{len(self.zipped_features_path)} != {self.labels}"
                     
    def get_zipped_path(self):
        zipped_features_path = []
        video_dict, audio_dict = {},{}
        for video_file in self.video_features_path:
            name = video_file.split('\\')[-1].split('@')[0]
            video_dict[name] = video_file
        for audio_file in self.audio_features_path:
            name = audio_file.split('\\')[-1].split('@')[0]
            audio_dict[name] = audio_file
            
        for name in video_dict:
            if name in audio_dict:
                zipped_features_path.append([video_dict[name], audio_dict[name]])
            else:
                zipped_features_path.append([video_dict[name], "[silent_video]"])
                self.silent_video_list.append(video_dict[name])

        self.zipped_features_path =  zipped_features_path
        self.audio_features_path = [feature[1] for feature in self.zipped_features_path]
        self.video_features_path = [feature[0] for feature in self.zipped_features_path]
        
    def map_label(self):
        return list(map(self.label_mapper,[i[0] for i in self.zipped_features_path]))

    def discard_silent_video(self):
        print("A new Dataset is returned after discarding silent video")
        before_discarding = len(self.zipped_features_path)
        print(f'Before discarding: {before_discarding} files were found')
        
        video_with_sound = [not re.search('silent_video', feature[1]) for feature in self.zipped_features_path]
        
        self.zipped_features_path = list(compress(self.zipped_features_path, video_with_sound))
        
        print(f'After discarding: {len(self.zipped_features_path)} files remained in this dataset')
        
        self.video_features_path = [feature[0] for feature in self.zipped_features_path]
        self.audio_features_path = [feature[1] for feature in self.zipped_features_path]
        self.labels = list(compress(self.labels, video_with_sound))
        self.silent_video_list = []

    def shuffle_dataset(self):
        self.zipped_features_path, self.audio_features_path, self.video_features_path, self.labels = \
            shuffle(self.zipped_features_path, self.audio_features_path, self.video_features_path, self.labels)
        
    def make_test_split(self, split = .1):
        split = int(len(self.zipped_features_path) * split)  if split < 1 else split
        test_dataset = copy.deepcopy(self)
        self.shuffle_dataset()

        self.zipped_features_path, test_dataset.zipped_features_path = self.zipped_features_path[:-split], self.zipped_features_path[-split:]
        self.video_features_path , test_dataset.video_features_path = self.video_features_path[:-split], self.video_features_path[-split:]
        self.audio_features_path , test_dataset.audio_features_path = self.audio_features_path[:-split], self.audio_features_path[-split:]
        self.labels , test_dataset.labels = self.labels[:-split], self.labels[-split:]
        
        return self, test_dataset
       
    def merge_dataset(datasets):
        """Discard silent video from individual datasets before merging them into one dataset
            else it might confilct with the label mapper
        """
        
        base_dataset = datasets[0]
        if len(datasets) > 1:
            for dataset in datasets[1:]:
                base_dataset.video_features_path.extend(dataset.video_features_path)
                base_dataset.audio_features_path.extend(dataset.audio_features_path)
                base_dataset.zipped_features_path.extend(dataset.zipped_features_path)
                base_dataset.labels.extend(dataset.labels)
                base_dataset.silent_video_list.extend(dataset.silent_video_list)
                
        assert len(base_dataset.zipped_features_path) == len(base_dataset.labels),\
            f"{len(base_dataset.zipped_features_path)} != {len(base_dataset.labels)}"
        print(f'After merging: {len(base_dataset.zipped_features_path)} files in this dataset')          
        return base_dataset
    
    
if __name__ == '__main__':
    rlvs = Dataset(video_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\rlvs\video\i3d.10s",
                   audio_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\rlvs\audio\yamnet",
                   label_mapper = LabelingPattern.rlvs,
                   discard_silent= True)
    
    my_dataset = Dataset(video_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\my_dataset\video\i3d.10s",
                   audio_features_path = r"F:\S-Home\ViolenceRecognizer\data\features\my_dataset\audio\yamnet",
                   label_mapper = LabelingPattern.my_dataset)
    my_dataset.get_zipped_path()
    combined_dataset = Dataset.merge_dataset([rlvs,my_dataset])#, discard_silent = True)
    
        
    


