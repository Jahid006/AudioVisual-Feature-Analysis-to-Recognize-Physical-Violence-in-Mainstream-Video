import numpy as np
import os, tqdm
import cfg
import video_utils
#from .utility import ask_for_confirmation
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def ask_for_confirmation(msg=''):
    if True:#CHECK_CONFIRMATION:
        if(input(f'\n{msg} continue...y/n: ').lower()!='y'):
            if (input('We are exiting....y/n: ').lower()=='y'):
                exit()
    else: print(f'\n{msg}')

def get_backbone(backbone):
    import tensorflow as tf
    if backbone == 'DenseNet201':
        from tensorflow.keras.applications.densenet import preprocess_input
        backbone = tf.keras.applications.DenseNet201(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        backbone = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",pooling='avg')
        
    return backbone, preprocess_input

def image_based_extractor(extractor_config):
    import tensorflow as tf, decord
    decord.bridge.set_bridge('tensorflow')

    backbone, preprocess_input = get_backbone(extractor_config.backbone_name)
    
    for video_file_name in tqdm.tqdm(extractor_config.video_file_list):
        frames = video_utils.ExtractFeatureDECORD(video_file_name,
                                                    extractor_config.frames,
                                                    extractor_config.width,
                                                    extractor_config.height
                                                    )
        frames = tf.cast(frames, tf.float32)
        #frames = tf.Tensor(frames, dtype=tf.float32)
        features = np.array(backbone.predict(preprocess_input(frames)))
        np.save(extractor_config.get_save_dir(video_file_name,
                                              extractor_config.backbone_name,
                                              extractor_config.frames,
                                              extractor_config.version),                          
                features ,
                allow_pickle=True)
    
def mxnet_based_extractor(extractor_config):
    from mxnet_based_feature_extractor import feature_extractor
    
    feature_extractor(extractor_config)

if __name__ == "__main__":
    
    IF_MXNET = True
    
    if IF_MXNET :
        extractor_config = cfg.CONFIG_MXNET
        ask_for_confirmation('You must configure the mxnet config file first.You have done it, right ?')
        mxnet_based_extractor(extractor_config)
        
    else:
        extractor_config = cfg.CONFIG
        ask_for_confirmation('You must configure the config file first.You have done it, right ?')
        image_based_extractor(extractor_config)