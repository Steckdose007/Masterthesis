
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import pickle
from plotting import plot_mel_spectrogram
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from audiodataloader import AudioDataLoader, AudioSegment
import random
import cv2
from tqdm import tqdm  
import os
from data_augmentation import apply_augmentation
import torchvision.transforms as transforms
import torch.nn.functional as F
from create_fixed_list import TrainSegment
from sklearn.preprocessing import MinMaxScaler
from math import exp, sqrt, pi

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  
    label_path: str
    path: str

class FixedListDataset(Dataset):
    def __init__(self, audio_segments: List[AudioSegment]):
        """
        Custom dataset for audio segments, prepares them for use in the CNN model.
        
        Parameters:
        - audio_segments: A list of AudioSegment objects.
        - target_length: The fixed length for padding/truncation.
        """
        self.audio_segments = audio_segments
        self.transforms  = transforms.Compose([
                            #transforms.RandomRotation(degrees=(-15, 15)),  # Rotate within -15 to 15 degrees
                            #transforms.RandomResizedCrop(size=(128, 256), scale=(0.8, 1.0)),  # Random crop and resize
                            #transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
                            #transforms.RandomVerticalFlip(p=0.2),  # 20% chance of vertical flip
                            #stt+mel
                            #transforms.Normalize(mean=[0.27651408,  0.30779094070238355    ], std=[0.13017927,0.22442872857101556])  
                            #mel
                            transforms.Normalize(mean=[0.30779094070238355  ], std=[0.22442872857101556])  
                            #mfcc
                            #transforms.Normalize(mean=[0.7017749593696507], std=[0.04512508429596211])  
                            #stt
                            #transforms.Normalize(mean=[0.27651408 ], std=[0.13017927])  
                            ])
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        object = self.audio_segments[idx]

        
        # ===== Process STT Feature =====
        stt_feature = object.mel#stt.detach().cpu().numpy()[0]  
        stt_resized = cv2.resize(stt_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        stt_scaled = self.scaler.fit_transform(stt_resized)  # Scale the STT feature
        att_map = self.generate_attention_map(
                word=object.label_word,
                mel_shape=stt_scaled.shape,
                focus_phonemes=('s', 'z', 'x'),
                sigma_time=30.0,
                amplitude=1.0
            )
        print(object.label_word)
        plot_mel_and_attention(stt_resized, att_map)
        # # ===== Process MFCC (Mel) Feature =====
        # mel_feature = object.mel  
        # mel_resized = cv2.resize(mel_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        # mel_scaled = self.scaler.fit_transform(mel_resized)  # Scale the MFCC feature

        """ When stacked for mel + stt"""
        # # ===== Stack Features =====
        # stt_tensor = torch.tensor(stt_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        # mel_tensor = torch.tensor(mel_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        # stacked_features = torch.cat([stt_tensor, mel_tensor], dim=0)  # Shape: (2, 224, 224)
        #print(stacked_features.shape)
        label = 0
        label_str = object.label_path
        if label_str == "sigmatism":
             label = 1

        featur_tensor = torch.tensor(stt_scaled, dtype=torch.float32).unsqueeze(0)
        feature_normalized = self.transforms(featur_tensor) 
        return  feature_normalized,label


    def gaussian_1d(self, x, mu, sigma):
        """Compute 1D Gaussian value at x with mean mu and std sigma."""
        return (1.0 / (sigma * sqrt(2.0 * pi))) * exp(-0.5 * ((x - mu) / sigma) ** 2)


    def generate_attention_map(self,word: str,mel_shape: tuple,focus_phonemes=('s', 'z', 'x'),sigma_time=5.0,amplitude=1.0,extra_coverage=15):
        """
        Generate a 2D attention map (freq_bins x time_steps) for a given word.
        Places Gaussian bumps in the time dimension for each focus phoneme.

        If the focus phoneme is at the beginning or end of the word, this function
        adds extra coverage in the time dimension to reflect extended emphasis.

        Args:
            word (str): The input word (e.g., 'Sonne').
            mel_shape (tuple): (freq_bins, time_steps) of the mel spectrogram.
            focus_phonemes (tuple): Characters to focus on (e.g. ('s', 'z', 'x')).
            sigma_time (float): Standard deviation for the time-axis Gaussian.
            amplitude (float): Peak amplitude for each Gaussian.
            extra_coverage (int): Additional time steps added before the first focus 
                                phoneme or after the last one.

        Returns:
            np.ndarray: A 2D array (freq_bins, time_steps) representing the attention map.
        """
        freq_bins, time_steps = mel_shape
        attention_map = np.zeros((freq_bins, time_steps), dtype=np.float32)

        # For each character in the word, if it's a focus phoneme, create a Gaussian
        L = len(word)

        for i, ch in enumerate(word.lower()):
            if ch in focus_phonemes:
                # Determine the center time step for this character
                center_time = (i + 0.5) * time_steps / L
                # Shift if it's the first character (move left)
                if i == 0:
                    center_time += extra_coverage

                # Shift if it's the last character (move right)
                elif i == L - 1:
                    center_time -= extra_coverage
                # Create a 1D Gaussian across this extended range, but clamp indices
                for t in range(time_steps):
                    g = self.gaussian_1d(t, center_time, sigma_time)
                    attention_map[:, t] += amplitude * g


        if np.max(attention_map) > 0:
            attention_map /= np.max(attention_map)
        return attention_map
    
def plot_mel_and_attention(mel_spectrogram, attention_map):
    """
    Plots two subplots vertically: the mel spectrogram on the top 
    and the attention map on the bottom.
    
    Args:
        mel_spectrogram (np.ndarray): 2D array (freq_bins, time_steps).
        attention_map (np.ndarray):    2D array (freq_bins, time_steps).
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    axes[0].imshow(
        mel_spectrogram,
        aspect='auto',
        origin='lower',
        cmap='plasma'
    )
    axes[0].set_title("Mel Spectrogram")
    axes[0].set_ylabel("Frequency Bins")

    axes[1].imshow(
        attention_map,
        aspect='auto',
        origin='lower',
        cmap='plasma'
    )
    axes[1].set_title("Attention Map")
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Frequency Bins")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    with open("data_lists/mother_list.pkl", "rb") as f:
        data = pickle.load(f)
    # Create dataset 
    segments_test = FixedListDataset(data[:50])
    logits,label = segments_test[20] 
    print(type(logits),logits.shape) 
    resized_array = logits.squeeze().detach().numpy()

    # Plot the resized logits as an image
    # plt.figure(figsize=(10, 6))
    # plt.imshow(resized_array, aspect='auto', origin='lower', cmap='plasma')
    # plt.colorbar(label='Intensity')
    # plt.tight_layout()
    # plt.show()
    
    
    