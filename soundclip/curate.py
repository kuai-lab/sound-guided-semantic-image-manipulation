from glob import glob
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import parmap
import os

audio_lists = glob("./vggsound/*.wav")
data_length = len(audio_lists)

def func(idx):
    try:
        wav_name = audio_lists[idx]        
        
        name = wav_name.split("/")[-1].split(".")[0]
        path = f"./vggsound_curation/{name}"

        if not os.path.exists(path):
            y, sr = librosa.load(wav_name, sr=44100)
            audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1
            audio_inputs = np.array([audio_inputs])
            np.save(path, audio_inputs)
        os.remove(wav_name)
    except:
        print(wav_name)
    finally:
        return 0

result = parmap.map(func, range(data_length), pm_pbar=True, pm_processes=16)