import librosa
import numpy as np
import random
import librosa

import random


def add_gaussian_noise(audio_data,sample_rate, noise_level=0.01):
    noise = np.random.normal(0, noise_level, audio_data.shape)
    return audio_data + noise

def time_stretch(audio_data, sample_rate):
    """
    Stretch or compress the time of the audio without changing the pitch.
    """
    if audio_data.size >= 2048:
        stretch_factor = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio_data, rate= stretch_factor)
    return add_gaussian_noise(audio_data,sample_rate)
    
def pitch_shift(audio_data, sample_rate):
    """
    Shift the pitch of the audio up or down.
    """
    if audio_data.size >= 2048:
        n_steps = random.randint(-2, 2)  # Shift pitch by up to 2 semitones
        return librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)
    return add_gaussian_noise(audio_data,sample_rate)
    
def random_crop_pad(audio_data, sample_rate):
    """
    Randomly crop or pad the audio signal.
    """
    crop_start = random.randint(0, len(audio_data) // 10)  # Randomly crop up to 10% from the start
    crop_end = random.randint(len(audio_data) - len(audio_data) // 10, len(audio_data))  # Random crop at the end
    cropped_audio = audio_data[crop_start:crop_end]
    return cropped_audio


def apply_augmentation(audio_data, sample_rate):
        """
        Apply data augmentation to the audio signal.
        
        Parameters:
        - audio_data: Numpy array of the audio signal.
        - sample_rate: Sample rate of the audio signal.
        
        Returns:
        - Augmented audio data.
        """
        augmentations = [
            add_gaussian_noise,
            time_stretch,
            pitch_shift,
            random_crop_pad
        ]
        augmentation = random.choice(augmentations)
        return augmentation(audio_data, sample_rate)
   
