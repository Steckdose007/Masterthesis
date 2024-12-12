
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

class GradcamDataset(Dataset):
    def __init__(self, audio_segments: List[AudioSegment], mfcc_dict : dict, augment: bool):
        """
        Custom dataset for audio segments, prepares them for use in the CNN model.
        
        Parameters:
        - audio_segments: A list of AudioSegment objects.
        - target_length: The fixed length for padding/truncation.
        """
        self.augment = augment
        self.audio_segments = audio_segments
        self.target_length = mfcc_dict["target_length"]
        self.mfcc_dict = mfcc_dict

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        audio_data = segment.audio_data
        label = 0
        if(segment.label_path == "sigmatism"):
            label = 1
        if self.augment and random.random() < 0.8 and audio_data.size >= 2048:  # 80% chance of augmentation
            audio_data = apply_augmentation(audio_data, segment.sample_rate)
        mfcc = self.compute_mfcc_features(audio_data,segment.sample_rate,n_mfcc=self.mfcc_dict["n_mfcc"], n_mels=self.mfcc_dict["n_mels"],
                        frame_size=self.mfcc_dict["frame_size"], hop_size=self.mfcc_dict["hop_size"], n_fft=self.mfcc_dict["n_fft"])
        #self.plot_histograms(mfcc)
        normalized_mfcc = self.normalize_mfcc(mfcc)
        #self.plot_histograms(normalized_mfcc)
        mel_specto = self.compute_melspectogram_features(audio_data,segment.sample_rate, n_mels=self.mfcc_dict["n_mels"],
                        frame_size=self.mfcc_dict["frame_size"], hop_size=self.mfcc_dict["hop_size"], n_fft=self.mfcc_dict["n_fft"])
        
        padding_mel,mel_audio = self.pad_mfcc(mel_specto, self.target_length)
        padding, padded_audio = self.pad_mfcc(normalized_mfcc, self.target_length)
        
        # Convert to PyTorch tensor and add channel dimension for CNN
        # In raw mono audio, the input is essentially a 1D array of values (e.g., the waveform). 
        # However, CNNs expect the input to have a channel dimension, 
        # which is why we add this extra dimension.
        audio_tensor = torch.tensor(padded_audio, dtype=torch.float32).unsqueeze(0) 
        mel_tensor = torch.tensor(mel_audio, dtype=torch.float32).unsqueeze(0)
        print(segment.label)#e.g Sonne
        #print("MFCC size: ",padded_audio.shape)
        
        return audio_tensor, label,segment.label, segment.audio_data, padding, mel_tensor,padding_mel

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
        return (paddingY,paddingX),mfcc[:target_n_mfcc, :target_time_frames]
        
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
    audio_dataset = AudioSegmentDataset(words_segments, target_length_24kHz_MFCC, augment=False)
    mfcc_tensor, label = audio_dataset[507]  # Fetch the first sample
    print(f"Label: {'Sigmatism' if label == 1 else 'Normal'}")
    print(mfcc_tensor.shape)
    # Plot the MFCC
    plot_mfcc(mfcc_tensor)
    #compute_average_spectrum_from_dataset(audio_dataset)

    # # Create DataLoader
    # train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
    
    