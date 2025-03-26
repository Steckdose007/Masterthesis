
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
import torchvision.transforms as transforms
import torchaudio.transforms as T
from data_augmentation import apply_augmentation
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

@dataclass
class TrainSegment:
    start_time: float
    end_time: float
    mfcc: np.ndarray
    mel: np.ndarray
    stt: np.ndarray
    sample_rate: int
    label_word: str  #word for example "sonne"
    label_path: str # sigmatism or normal
    path: str # which file it is from
    augmented: str #if that was augmented with pitch/noise

class GradcamDataset(Dataset):
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
                            #transforms.Normalize(mean=[0.27651408,  0.30779094070238355    ], std=[0.13017927,0.22442872857101556]),  
                            #mel
                            transforms.Normalize(mean=[0.30779094070238355  ], std=[0.22442872857101556]),  
                            #mfcc
                            #transforms.Normalize(mean=[0.7017749593696507], std=[0.04512508429596211]),
                            #stt
                            #transforms.Normalize(mean=[0.27651408 ], std=[0.13017927])  ,
                            #T.TimeMasking(time_mask_param=30),  # Mask up to 30 time steps
                            #T.FrequencyMasking(freq_mask_param=30)  # Mask up to 15 frequency bins
                            ])
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        object = self.audio_segments[idx]

        
        # ===== Process STT Feature =====
        # stt_feature = object.stt.detach().cpu().numpy()[0]  
        # stt_resized = cv2.resize(stt_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        # stt_scaled = self.scaler.fit_transform(stt_resized)  # Scale the STT feature
        

        # ===== Process MFCC (Mel) Feature =====
        mel_feature = object.mel  
        mel_resized = cv2.resize(mel_feature, (224, 224), interpolation=cv2.INTER_LINEAR)
        mel_scaled = self.scaler.fit_transform(mel_resized)  # Scale the MFCC feature

        att_map = generate_attention_map(word=object.label_word,mel_shape=mel_scaled.shape,focus_phonemes=('s', 'z', 'x'),sigma_time=30.0,amplitude=1.0)
        att_tensor = torch.tensor(att_map, dtype=torch.float32).unsqueeze(0) 
        # print(object.label_word)
        # plot_mel_and_attention(stt_resized, att_map)
        """ When stacked for mel + stt"""
        # ===== Stack Features =====
        # stt_tensor = torch.tensor(stt_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        # mel_tensor = torch.tensor(mel_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 224, 224)
        # stacked_features = torch.cat([stt_tensor, mel_tensor], dim=0)  # Shape: (2, 224, 224)
        #print(stacked_features.shape)
        label = 0
        label_str = object.label_path
        if label_str == "sigmatism":
             label = 1

        featur_tensor = torch.tensor(mel_scaled, dtype=torch.float32).unsqueeze(0)
        feature_normalized = self.transforms(featur_tensor) 
        final_features = torch.cat([feature_normalized, att_tensor], dim=0)
        return  final_features, label, object.label_word
        
    def plot_histograms(self,mfcc):
        """
        To show the range of mfcc in order to analyse the 
        
        """
        mfcc_values = mfcc.flatten()
        plt.hist(mfcc_values, bins=100)
        plt.title("Histogram of MFCC Values")
        plt.xlabel(" MFCC Value")
        plt.ylabel("Frequency")
        plt.show()

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot(mfcc_values, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title("Boxplot of MFCC Values")
        plt.xlabel("MFCC Value")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()

def gaussian_1d( x, mu, sigma):
        """Compute 1D Gaussian value at x with mean mu and std sigma."""
        return (1.0 / (sigma * sqrt(2.0 * pi))) * exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_attention_map(word: str,mel_shape: tuple,focus_phonemes=('s', 'z', 'x'),sigma_time=5.0,amplitude=1.0,extra_coverage=15):
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
                    g = gaussian_1d(t, center_time, sigma_time)
                    attention_map[:, t] += amplitude * g


        if np.max(attention_map) > 0:
            attention_map /= np.max(attention_map)
        return attention_map

def compute_average_spectrum_from_dataset(dataset, target_sample_rate=44100):

    spectrums = []
    i=0
    for audio_tensor, _ in dataset:
        # Remove the channel dimension and convert to numpy
        audio = audio_tensor.squeeze(0).numpy()

        # Compute the FFT
        fft = np.fft.fft(audio)
        fft_magnitude = np.abs(fft)  # Magnitude of FFT
        
        # Frequency bins
        sample_rate = target_sample_rate  # Ensure sample rate is consistent
        frequencies = np.fft.fftfreq(len(fft), 1 / sample_rate)
        
        # Only take the positive frequencies
        half_spectrum = len(fft_magnitude) // 2
        fft_magnitude = fft_magnitude[:half_spectrum]
        
        spectrums.append(fft_magnitude)
        i +=1
        if(i>1000):
            break
    # Stack and compute the average spectrum
    spectrums = np.array(spectrums)
    average_spectrum = np.mean(spectrums, axis=0)
    frequencies = frequencies[:half_spectrum]
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, average_spectrum, color='blue', label="Average Spectrum")
    plt.title("Average Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()
    plt.show()

def plot_frequ_and_acc():
    sampling_rates = [8000, 16000, 24000, 32000, 44100]
    accuracies = [0.586, 0.586, 0.593, 0.587, 0.570]  # Replace with your actual accuracies
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(rate) + " Hz" for rate in sampling_rates], accuracies, color='skyblue', edgecolor='black')

    # Add accuracy values inside the bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height() - 0.05,  # Slightly below the top of the bar
                f"{accuracy:.3f}", 
                ha='center', va='bottom', fontsize=12, color='black', weight='bold')

    # Add labels and title
    plt.title("Sampling Frequency vs. Accuracy", fontsize=16)
    plt.xlabel("Sampling Frequency (Hz)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1)  # Adjust depending on your accuracy range
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_mfcc(mfcc_tensor, sample_rate= 24000, title="MFCC"):

    """
    Plot the MFCC features as a heatmap.

    Parameters:
    - mfcc_tensor: The MFCC features (2D numpy array or PyTorch tensor).
    - sample_rate: The sample rate of the audio.
    - title: Title for the plot.
    """
    # Convert tensor to numpy array if necessary
    if isinstance(mfcc_tensor, torch.Tensor):
        mfcc_tensor = mfcc_tensor.squeeze().numpy()

    #mfcc_tensor = (mfcc_tensor - np.mean(mfcc_tensor)) / np.std(mfcc_tensor)
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.imshow(mfcc_tensor, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(label="MFCC")
    plt.title("MFCC")
    plt.xlabel("Frames")
    plt.ylabel("MFCC Coefficients")
    plt.show()

if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    
    # words_segments = loader.create_dataclass_words()
    # loader.save_segments_to_pickle(words_segments, "words_segments.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    #visualize_augmentations(words_segments[1].audio_data, words_segments[1].sample_rate)
    #print(np.shape(words_segments))
    # target_length = int(1.2*11811)    
    target_length_24kHz_MFCC = int(256)
    audio_dataset = GradcamDataset(words_segments, target_length_24kHz_MFCC, augment=False)
    mfcc_tensor, label = audio_dataset[507]  # Fetch the first sample
    print(f"Label: {'Sigmatism' if label == 1 else 'Normal'}")
    print(mfcc_tensor.shape)
    # Plot the MFCC
    plot_mfcc(mfcc_tensor)
    #compute_average_spectrum_from_dataset(audio_dataset)

    # # Create DataLoader
    # train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    
    