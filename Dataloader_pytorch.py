
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
    def __init__(self, audio_segments: List[AudioSegment], target_length: int, augment: bool):
        """
        Custom dataset for audio segments, prepares them for use in the CNN model.
        
        Parameters:
        - audio_segments: A list of AudioSegment objects.
        - target_length: The fixed length for padding/truncation.
        """
        self.augment = augment
        self.audio_segments = audio_segments
        self.target_length = target_length

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        audio_data = segment.audio_data
        label = 0
        if(segment.label_path == "sigmatism"):
            label = 1
        # Apply augmentation to a subset of samples (e.g., 50% chance)
        if self.augment and random.random() < 0.8:  # 80% chance of augmentation
            audio_data = apply_augmentation(audio_data, segment.sample_rate)
        padded_audio = self.pad_audio(audio_data, self.target_length)
        
        # Convert to PyTorch tensor and add channel dimension for CNN
        # In raw mono audio, the input is essentially a 1D array of values (e.g., the waveform). 
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
        
    
def visualize_augmentations(audio_data, sample_rate):
    """
    Visualize the effects of data augmentation on audio data.
    
    Parameters:
    - audio_data: The original audio signal (numpy array).
    - sample_rate: The sample rate of the audio signal.
    """
    # Define augmentations
    def add_gaussian_noise(audio_data, noise_level=0.02):
        noise = np.random.normal(0, noise_level, audio_data.shape)
        return audio_data + noise

    def time_stretch(audio_data, sample_rate, stretch_factor=1.2):
        return librosa.effects.time_stretch(audio_data, rate= stretch_factor)

    def pitch_shift(audio_data, sample_rate, n_steps=2):
        return librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)

    def random_crop_pad(audio_data, target_length):
        if len(audio_data) < target_length:
            # Pad with zeros
            return np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        else:
            # Randomly crop
            crop_start = np.random.randint(0, len(audio_data) - target_length)
            return audio_data[crop_start:crop_start + target_length]

    # Target length for cropping/padding
    target_length = int(0.8 * len(audio_data))  # 80% of original length

    # Apply augmentations
    augmentations = [
        ("Original", audio_data),
        ("Gaussian Noise", add_gaussian_noise(audio_data)),
        ("Time Stretch (slower)", time_stretch(audio_data, sample_rate, stretch_factor=0.8)),
        ("Pitch Shift (+2 semitones)", pitch_shift(audio_data, sample_rate, n_steps=2)),
        ("Random Crop/Pad", random_crop_pad(audio_data, target_length))
    ]

    # Plot the augmentations
    plt.figure(figsize=(12, 8))
    for i, (title, augmented_data) in enumerate(augmentations):
        plt.subplot(len(augmentations), 1, i + 1)
        plt.plot(augmented_data, alpha=0.7, label=title)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid()

    plt.tight_layout()
    plt.show()

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

if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    
    # words_segments = loader.create_dataclass_words()
    # loader.save_segments_to_pickle(words_segments, "words_segments.pkl")
    words_segments = loader.load_segments_from_pickle("all_words_segments.pkl")
    visualize_augmentations(words_segments[1].audio_data, words_segments[1].sample_rate)
    #print(np.shape(words_segments))
    # target_length = int(1.2*11811)    
    # audio_dataset = AudioSegmentDataset(words_segments, target_length)
    # compute_average_spectrum_from_dataset(audio_dataset)

    # # Create DataLoader
    # train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    
    