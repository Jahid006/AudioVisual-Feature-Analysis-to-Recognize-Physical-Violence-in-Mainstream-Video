%reset -f
import gc
import tensorflow_hub as hub,tensorflow as tf
import tensorflow_io as tfio
import tqdm,numpy as np,os,tqdm,glob
from IPython.display import clear_output


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

@tf.function
def load_wav_16k_mono(filename):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

start = 0
class_ = "notPhysicalViolence"
audio_data = glob.glob(f'audio\{class_}\*')
os.makedirs(f"audio_embedding/{class_}" , exist_ok=True)  

'''import multiprocessing
p = multiprocessing.Pool(3)
embeddingss = p.map(save_embedding, audio_data[:6])'''


def save_embedding(audio_path):
    """ save the embedding of the audio file """
    gg, embeddings, kk = yamnet_model(load_wav_16k_mono(audio_path))  
    name = audio_path.split('\\')[-1][:-4]+'.wav'
    
    sname = os.path.join(f'audio_embedding/{class_}',name[:-4]+'.npy')
    print(sname)
    embeddings = np.max(embeddings,axis=0)*.25+np.mean(embeddings,axis=0)*.75 
    embeddings = embeddings.reshape(-1)
    print(embeddings.shape)

    np.save(sname,embeddings , allow_pickle=True)
    
    
for i in tqdm.tqdm(audio_data[:]):
    save_embedding(i)
    clear_output()