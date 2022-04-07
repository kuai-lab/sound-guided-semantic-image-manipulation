from librosa.core import audio
from scipy.signal import waveforms
from torch.utils.data.dataset import Dataset
import librosa
from glob import glob
import cv2
import numpy as np
import torch 
import random
import pandas as pd
import time
import clip
from textaugment import EDA
import nltk
import pickle
from PIL import Image
import os

nltk.download("stopwords")
nltk.download("wordnet")

class VggsoundCurationDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("./vggsound_curation/*.npy")
        self.time_length = 864
        self.n_mels = 128
        self.text_aug = EDA()
        self.width_resolution = 512

    def __getitem__(self, idx):
        wav_name = self.audio_lists[idx]
            
        audio_inputs = np.load(wav_name, allow_pickle=True)

        text_prompt = wav_name.split("/")[-1].split("_")[0]
        c, h, w = audio_inputs.shape

        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
       
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
            
        audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)
            
        audio_inputs = torch.from_numpy(audio_inputs).float()
        audio_aug = torch.from_numpy(audio_aug).float()

        text_prompt = self.text_aug.synonym_replacement(text_prompt)
        text_prompt = self.text_aug.random_swap(text_prompt)
        text_prompt = self.text_aug.random_insertion(text_prompt)
            
        return audio_inputs, audio_aug, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        return spec

    def __len__(self):
        return len(self.audio_lists)

class AudiosetBalancedCurationDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("./audioset_balanced_curation/*.npy")
        self.time_length = 864
        self.n_mels = 128
        self.text_aug = EDA()
        self.width_resolution = 512
        self.labels = []
        for idx, path in enumerate(self.audio_lists):
            label = self.audio_lists[idx].split("/")[-1].split("_")[1]
            self.labels.append(label)
            
    def __getitem__(self, idx):
        wav_name = self.audio_lists[idx]
            
        audio_inputs = np.load(wav_name, allow_pickle=True)

        text_prompt = wav_name.split("/")[-1].split("_")[1]
        c, h, w = audio_inputs.shape
        if w >= self.time_length:
            j = random.randint(0, w-self.time_length)
            audio_inputs = audio_inputs[:,:,j:j+self.time_length]
        elif w < self.time_length:
            zero = np.zeros((1, self.n_mels, self.time_length))
            j = random.randint(0, self.time_length - w - 1)
            zero[:,:,j:j+w] = audio_inputs[:,:,:w]
            audio_inputs = zero
            
        audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))
            
        audio_aug = self.spec_augment(audio_inputs)
        audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
        audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)

        audio_inputs = torch.from_numpy(audio_inputs).float()
        audio_aug = torch.from_numpy(audio_aug).float()
            
        text_prompt = self.text_aug.synonym_replacement(text_prompt)
        text_prompt = self.text_aug.random_swap(text_prompt)
        return audio_inputs, audio_aug, text_prompt


    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        
        return spec
    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.audio_lists)


class AudiosetUnbalancedCurationDataset(Dataset):
    def __init__(self):
        self.audio_lists = glob("./audioset_unbalanced_curation/*.npy")
        self.time_length = 864
        self.n_mels = 128
        self.text_aug = EDA()
        self.width_resolution = 512

        self.labels = pd.read_csv("./class_labels_indices.csv", sep=",")
        self.dataframe = pd.read_csv("./unbalanced_train_segments.csv", sep=" ")

        with open("./cache.pkl", "rb") as f:
            self.cache = pickle.load(f)

    def __getitem__(self, idx):
        try:
            wav_name = self.audio_lists[idx]
            
            audio_inputs = np.load(wav_name)
            audio_key = wav_name.split("/")[-1].split(".")[0][1:] + ","

            
            text_prompt = self.cache[audio_key]

            c, h, w = audio_inputs.shape
            if w >= self.time_length:
                j = random.randint(0, w-self.time_length)
                audio_inputs = audio_inputs[:,:,j:j+self.time_length]
            elif w < self.time_length:
                zero = np.zeros((1, self.n_mels, self.time_length))
                j = random.randint(0, self.time_length - w - 1)
                zero[:,:,j:j+w] = audio_inputs[:,:,:w]
                audio_inputs = zero

        except Exception as e:
            audio_inputs = np.zeros((1, self.n_mels, self.time_length)) 
            text_prompt = "no sound"

        finally:
            audio_inputs = cv2.resize(audio_inputs[0], (self.n_mels, self.width_resolution))            
            audio_aug = self.spec_augment(audio_inputs)
            audio_inputs = audio_inputs.reshape(-1, self.n_mels, self.width_resolution)
            audio_aug = audio_aug.reshape(-1, self.n_mels, self.width_resolution)

            audio_inputs = torch.from_numpy(audio_inputs).float()
            audio_aug = torch.from_numpy(audio_aug).float()
            
            text_prompt = self.text_aug.synonym_replacement(text_prompt)
            text_prompt = self.text_aug.random_swap(text_prompt)
            return audio_inputs, audio_aug, text_prompt

    def spec_augment(self, spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
            
            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)
            
            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0
        
        return spec

    def __len__(self):
        return len(self.audio_lists)


if __name__ == "__main__":

    datasets = AudiosetUnbalancedCurationDataset()
    start = time.time()
    print(datasets[80])
    print(len(datasets), datasets[1090][0].size(), datasets[1090][1])
    print(time.time() - start)
