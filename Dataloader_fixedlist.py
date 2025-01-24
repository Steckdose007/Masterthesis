
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
                            transforms.Normalize(mean=[0.27651408,  0.30779094070238355    ], std=[0.13017927,0.22442872857101556])  
                            ])
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        object = self.audio_segments[idx]

        
        # ===== Process STT Feature =====
        stt_feature = object.stt.detach().cpu().numpy()[0]  # Assuming STT feature is available
        stt_resized = cv2.resize(stt_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        stt_scaled = self.scaler.fit_transform(stt_resized)  # Scale the STT feature

        # ===== Process MFCC (Mel) Feature =====
        mel_feature = object.mel  # Assuming `mel` is your MFCC feature
        mel_resized = cv2.resize(mel_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        mel_scaled = self.scaler.fit_transform(mel_resized)  # Scale the MFCC feature

        # ===== Stack Features =====
        stt_tensor = torch.tensor(stt_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        mel_tensor = torch.tensor(mel_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        stacked_features = torch.cat([stt_tensor, mel_tensor], dim=0)  # Shape: (2, 224, 224)

        label = 0
        label_str = object.label_path
        if label_str == "sigmatism":
             label = 1
        #print("processed",processed_object.shape)
        featur_tensor = torch.tensor(stacked_features, dtype=torch.float32).unsqueeze(0)
        feature_normalized = self.transforms(featur_tensor) 
        #print(transformed_mfcc.shape)
        return  feature_normalized,label



if __name__ == "__main__":

    with open("data_lists/mother_list.pkl", "rb") as f:
        data = pickle.load(f)
    # Create dataset 
    segments_test = FixedListDataset(data[:10])
    logits,label = segments_test[5] 
    print(type(logits),logits.shape) 
    resized_array = logits.squeeze().detach().numpy()

    # Plot the resized logits as an image
    plt.figure(figsize=(10, 6))
    plt.imshow(resized_array, aspect='auto', origin='lower', cmap='plasma')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()
    
    
    