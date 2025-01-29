
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
                            #transforms.ToTensor(),
                            #transforms.RandomRotation(degrees=(-15, 15)),  # Rotate within -15 to 15 degrees
                            #transforms.RandomResizedCrop(size=(128, 256), scale=(0.8, 1.0)),  # Random crop and resize
                            #transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
                            #transforms.RandomVerticalFlip(p=0.2),  # 20% chance of vertical flip
                            transforms.Normalize(mean=[0.5595994632236891], std=[0.08789908058262823])  # Normalize between -1 and 1
])

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        object = self.audio_segments[idx]

        heatmap = object["heatmap"]

        time_steps, vocab_size = heatmap.shape
    
        if time_steps < 84:
            # Amount to pad in the 'time_steps' dimension
            pad_amount = 84 - time_steps
            heatmap = F.pad(heatmap, pad=(0, 0, 0, pad_amount), mode='constant', value=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        #print(heatmap.shape)
        #then resize both
        heatmap = F.interpolate(heatmap,size = (224, 224)).squeeze(0)
        heatmap = (heatmap - (-25.749231)) / 43.0570068459375
        label = 0
        #print(heatmap.shape)
        label_str = object["label_path"]
        
        if label_str == "sigmatism":
             label = 1
        #print("processed",processed_object.shape)

        heatmap = self.transforms(heatmap)
        #print(transformed_mfcc.shape)
        return  heatmap,label




if __name__ == "__main__":

    with open("STT_csv\per_word_auc_values.pkl", "rb") as f:
        data = pickle.load(f)
    # Create dataset 
    segments_test = FixedListDataset(data)
    logits,label = segments_test[10] 
    print(type(logits)) 
    resized_array = logits.squeeze().detach().numpy()

    # Plot the resized logits as an image
    plt.figure(figsize=(10, 6))
    plt.imshow(resized_array, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Logit Intensity')
    plt.title("Resized Logits Visualization")
    plt.xlabel("Feature Dimension (Vocab Size)")
    plt.ylabel("Time Steps")
    plt.tight_layout()
    plt.show()
    
    
    