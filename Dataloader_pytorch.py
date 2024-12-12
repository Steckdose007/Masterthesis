
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
import random
import cv2
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

class AudioSegmentDataset(Dataset):
    def __init__(self, audio_segments: List[AudioSegment],phones_segments:List[AudioSegment], mfcc_dict : dict, augment: bool):
        """
        Custom dataset for audio segments, prepares them for use in the CNN model.
        
        Parameters:
        - audio_segments: A list of AudioSegment objects.
        - target_length: The fixed length for padding/truncation.
        """
        self.augment = augment
        self.phones = phones_segments
        self.audio_segments = audio_segments
        self.target_length = mfcc_dict["target_length"]
        self.mfcc_dict = mfcc_dict

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        phones = self.find_pairs(segment)
        hop_length = int(self.mfcc_dict["hop_size"] * segment.sample_rate)
        scaling=24000/44100
        frames_with_phone = []
        #get frames in the mfcc in which the phone is.
        for p in phones:
            # Adjust phone start and end times relative to the word start (in seconds)
            frame_start = int(((p.start_time - segment.start_time)*scaling) / hop_length)
            frame_end = int(((p.end_time - segment.start_time)*scaling) / hop_length)
            frames_with_phone.append((frame_start,frame_end))

        audio_data = segment.audio_data
        label = 0
        if(segment.label_path == "sigmatism"):
            label = 1
        if self.augment and random.random() < 0.8 and audio_data.size >= 2048:  # 80% chance of augmentation
            audio_data = apply_augmentation(audio_data, segment.sample_rate)
            
        mfcc = self.compute_mfcc_features(audio_data,segment.sample_rate,n_mfcc=self.mfcc_dict["n_mfcc"], n_mels=self.mfcc_dict["n_mels"],
                                           frame_size=self.mfcc_dict["frame_size"], hop_size=self.mfcc_dict["hop_size"], n_fft=self.mfcc_dict["n_fft"])
        normalized_mfcc = self.normalize_mfcc(mfcc)
        #mel_specto = self.compute_melspectogram_features(audio_data,segment.sample_rate, n_mels=self.mfcc_dict["n_mels"],
                                           #frame_size=self.mfcc_dict["frame_size"], hop_size=self.mfcc_dict["hop_size"], n_fft=self.mfcc_dict["n_fft"])
        
        resized_mfcc = self.extract_and_resize_mfcc(normalized_mfcc, frames_with_phone, target_size=(224, 224))
        #padded_audio = self.pad_mfcc(normalized_mfcc, self.target_length)
        
        # Convert to PyTorch tensor and add channel dimension for CNN
        # In raw mono audio, the input is essentially a 1D array of values (e.g., the waveform). 
        # However, CNNs expect the input to have a channel dimension, 
        # which is why we add this extra dimension.
        audio_tensor = torch.tensor(resized_mfcc, dtype=torch.float32).unsqueeze(0) 

        
        return audio_tensor, label

    def compute_mfcc_features(self,signal, sample_rate, n_mfcc=128, n_mels=128, frame_size=25.6e-3, hop_size=5e-3, n_fft=2048):
        try:
            # Convert frame and hop size from seconds to samples
            frame_length = int(frame_size * sample_rate)
            hop_length = int(hop_size * sample_rate)
            
            # Compute the static MFCCs using librosa's mfcc function
            mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, 
                                        n_fft=n_fft, hop_length=hop_length, win_length=frame_length, n_mels=n_mels)
            
            # Compute the first-order difference (Delta MFCCs) using a 5-frame window
            mfcc_delta = librosa.feature.delta(mfccs, width=5)
            
            # Compute the second-order difference (Delta-Delta MFCCs)
            #mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=3)
            
            # Concatenate static, delta, and delta-delta features to form a 24-dimensional feature vector per frame
            mfcc_features = np.concatenate([mfccs, mfcc_delta], axis=0)
            
            return mfcc_features
        except:
            print("ERROR: ",np.shape(signal), signal.size)

    def compute_melspectogram_features(self,signal, sample_rate=24000, n_mels=128, frame_size=25.6e-3, hop_size=5e-3, n_fft=2048):
        # Convert frame and hop size from seconds to samples
        frame_length = int(frame_size * sample_rate)
        hop_length = int(hop_size * sample_rate)   
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,win_length=frame_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mel_spectrogram_db_normalized = (mel_spectrogram_db + 80) / 80
        normalized_spectrogram = (mel_spectrogram_db_normalized - 0.485) / 0.229
        return normalized_spectrogram
    
    def normalize_mfcc(self,mfcc):
        """
        https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
        The images have to be loaded in to a range of [0, 1] 
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        """
        # Step 1: Scale to [0, 1] using min-max normalization
        mfcc_min = mfcc.min()
        mfcc_max = mfcc.max()
        mfcc_scaled = (mfcc - mfcc_min) / (mfcc_max - mfcc_min)

        # Step 2: Normalize using mean and std for the first channel
        mean = [0.485]  # Only the first channel is relevant for MFCC input
        std = [0.229]
        
        mfcc_normalized = (mfcc_scaled - mean[0]) / std[0]

        return mfcc_normalized
    
    #target frames was found out impirical.
    def pad_mfcc(self, mfcc, target_time_frames = 224,target_n_mfcc=224):        
        """
        Pad the audio signal to a fixed target length.
        If the audio is shorter, pad with zeros. If longer, truncate.
        
        Parameters:
        - audio_data: Numpy array of the audio signal.
        - target_length: Desired length in samples.
        
        Returns:
        - Padded or truncated audio data.
        """
        n_mfcc, time_frames = mfcc.shape

        # Compute padding or truncation for both axes
        paddingY = max(0, target_n_mfcc - n_mfcc)
        paddingX = max(0, target_time_frames - time_frames)

        # Pad if smaller
        if paddingY > 0 or paddingX > 0:
            mfcc = np.pad(
                mfcc,
                ((0, paddingY), (0, paddingX)),  # Pad both axes
                mode='constant'
            )

        # Truncate if larger
        return mfcc[:target_n_mfcc, :target_time_frames]

    def extract_and_resize_mfcc(self,mfcc, frames_with_phone, target_size=(224, 224)):
        
        # Extract only the frames with the phone
        extracted_mfcc = []
        for frame_start, frame_end in frames_with_phone:
            extracted_mfcc.append(mfcc[:, frame_start:frame_end])  # Append the relevant frames
        
        # Concatenate the extracted frames along the time axis
        if len(extracted_mfcc) > 0:
            extracted_mfcc = np.concatenate(extracted_mfcc, axis=1)
        else:
            raise ValueError("No frames found for the specified phone.")

        # Resize the extracted MFCC to the target size
        resized_mfcc = cv2.resize(extracted_mfcc, target_size, interpolation=cv2.INTER_LINEAR)

        return resized_mfcc

    def find_pairs(self,segment,phones_segments):

        phones =["z","s","Z","S","ts"]
        phones_list = []
       
        if(phones_segments):
            for phone in phones_segments:
                if (phone.label in phones and
                    phone.path == segment.path and
                    phone.sample_rate == segment.label):
                    phones_list.append(phone)
        return  phones_list

        


if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    
    # words_segments = loader.create_dataclass_words()
    # loader.save_segments_to_pickle(words_segments, "words_segments.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    #visualize_augmentations(words_segments[1].audio_data, words_segments[1].sample_rate)
    #print(np.shape(words_segments))
    # target_length = int(1.2*11811)    
    target_length_24kHz_MFCC = int(256)
    audio_dataset = AudioSegmentDataset(words_segments, target_length_24kHz_MFCC, augment=False)
    mfcc_tensor, label = audio_dataset[507]  # Fetch the first sample
    print(f"Label: {'Sigmatism' if label == 1 else 'Normal'}")
    print(mfcc_tensor.shape)
    # Plot the MFCC
    plot_mfcc(mfcc_tensor)
    #compute_average_spectrum_from_dataset(audio_dataset)

    # # Create DataLoader
    # train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    
    