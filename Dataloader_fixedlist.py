
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

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        processed_object, label = self.audio_segments[idx]
        

        return processed_object, label



if __name__ == "__main__":

    with open("segments_train.pkl", "rb") as f:
        data = pickle.load(f)
    # Create dataset 
    segments_test = FixedListDataset(data)
    print(len(segments_test))
    print(segments_test[0][0].shape)
    
    
    