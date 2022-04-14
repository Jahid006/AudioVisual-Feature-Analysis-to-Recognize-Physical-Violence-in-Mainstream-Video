import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers

import glob
import numpy as np

from config import INPUT_DIM, MODEL_PATH, LOG_PATH, IF_MXNET_MODEL, INPUT_DIM, LEARNING_RATE, EPOCHS
from utility import Exit


def exponential_decay(epoch,lr):
    """exponential decay learning rate"""
    lr = lr * 0.1**(epoch/10)#lr * np.exp(-0.1 * epoch/3)
    return lr

def scheduler(epoch, lr):
    #scheduler = lambda epoch,lr: lr *.5 if epoch>5 else lr
    if epoch < 5: return lr
    else: return (EPOCHS-epoch)*LEARNING_RATE/EPOCHS

def schedulerV2(epoch, lr):
    if epoch < 5: return lr
    #else:         return max(LEARNING_RATE - LEARNING_RATE*.1*(epoch//3), LEARNING_RATE*.001)
    else:         (LEARNING_RATE-LEARNING_RATE*.01)*0.85**((epoch+1)//5) + LEARNING_RATE*.01

def get_callbacks():
    """
    Models Callbacks:
    min_delta : minimum change in the monitored quantity to qualify as an improvement,i.e. an absolute change of less than min_delta, will count as no improvement.
    patience : number of epochs with no improvement after which training will be stopped.
    """
    #checkpoint = tf.keras.callbacks.ModelCheckpoint('violenceNet.h5',save_best_only=True)
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=10,min_delta= .001,mode='auto',monitor="val_loss", restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(filepath = MODEL_PATH+'\model_{epoch:03d}_{val_loss:.3f}_{val_accuracy:.3f}_.h5'),
                    tf.keras.callbacks.TensorBoard(log_dir=LOG_PATH),
                    tf.keras.callbacks.LearningRateScheduler(scheduler),]
    
    return my_callbacks

def get_model(dimension = 512,summary = False,input_shape = INPUT_DIM):
    
    FeatureInput = Input(shape=input_shape)
    Average_feature = layers.Flatten()(FeatureInput) if  IF_MXNET_MODEL \
                        else tf.reduce_max(FeatureInput,-2)*.25 + tf.reduce_mean(FeatureInput,-2)*.75
    
    AudioInput =   Input(shape=(1024))
    audio_ = layers.Dense(dimension//4,kernel_initializer='normal', activation='linear')(AudioInput)
    audio_ = layers.BatchNormalization()(audio_)
    audio_ = layers.Activation(tf.nn.relu)(audio_)
    audio_ = layers.Dropout(.5)(audio_)
    
    video_ = layers.Flatten()(Average_feature)
    video_ = layers.Dense(dimension//2,kernel_initializer='normal', activation='linear')(video_)
    video_ = layers.BatchNormalization()(video_)
    video_ = layers.Activation(tf.nn.relu)(video_)
    video_ = layers.Dropout(.5)(video_)
    
    combined = layers.concatenate([video_,audio_],axis=1)
    
    output = layers.Dense(1,kernel_initializer='normal', activation='sigmoid',name = 'classifier')(combined)
    
    model = Model(inputs= [FeatureInput,AudioInput], outputs = output)
    model.compile(optimizer=tf.optimizers.Nadam(learning_rate=LEARNING_RATE), loss='binary_crossentropy',metrics=['accuracy'])
    
    if summary:print(model.summary())
    
    return model

def get_best_model(history):
    """get the best model from history"""
    best_epoch = np.argmin(history['val_loss'])
    return history['model'][best_epoch]

def get_best_model_from_storage(path = MODEL_PATH,based_on_val_loss = True):
    
    models_ = glob.glob(path+'\*.h5')
    if len(models_) == 0: return None
    if based_on_val_loss:best_model = np.argmin([float(i.split('\\')[-1].split('_')[-3]) for i in models_])
    else:                best_model = np.argmax([float(i.split('\\')[-1].split('_')[-2]) for i in models_])
    
    
    print(f'best model is {models_[best_model]}')
    return models_[best_model]