
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np
import pandas as pd
from numba import jit
import torch
from torch.utils.data import Dataset, DataLoader
from audiodataloader import AudioDataLoader, AudioSegment


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  
    label_path: str

class AudioSegmentDataset(Dataset):
    def __init__(self, audio_segments: List[AudioSegment], target_length):
        """
        Custom dataset for audio segments, prepares them for use in the CNN model.
        
        Parameters:
        - audio_segments: A list of AudioSegment objects.
        - target_length: The fixed length for padding/truncation.
        """
        self.audio_segments = audio_segments
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        audio_data = segment.audio_data
        label = segment.label_path  # Assuming the label is an integer or encoded
        padded_audio = self.pad_audio(audio_data, self.target_length)
        
        # Convert to PyTorch tensor and add channel dimension for CNN
        #In raw mono audio, the input is essentially a 1D array of values (e.g., the waveform). 
        # However, CNNs expect the input to have a channel dimension, 
        # which is why we add this extra dimension.
        audio_tensor = torch.tensor(padded_audio, dtype=torch.float32).unsqueeze(0)  

        return audio_tensor, label

    def pad_audio(self,audio_data, target_length):
        """
        Pad the audio signal to a fixed target length.
        If the audio is shorter, pad with zeros. If longer, truncate.
        
        Parameters:
        - audio_data: Numpy array of the audio signal.
        - target_length: Desired length in samples.
        
        Returns:
        - Padded or truncated audio data.
        """
        if len(audio_data) < target_length:
            # Pad with zeros if the audio is too short
            return np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        else:
            # Truncate if the audio is too long
            return audio_data[:target_length]


if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    
    #words_segments = loader.create_dataclass_words()
    # loader.save_segments_to_pickle(words_segments, "words_segments.pkl")
    words_segments = loader.load_segments_from_pickle("words_segments.pkl")
    print(np.shape(words_segments))
    target_length = 291994  
    audio_dataset = AudioSegmentDataset(words_segments, target_length)

    # Create DataLoader
    train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    