
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from collections import defaultdict
import plotting



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
        MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    
    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        segment = self.audio_segments[idx]
        audio_data = segment.audio_data
        label = 0
        if(segment.label_path == "sigmatism"):
            label = 1
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits  = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        resized_logits = F.interpolate(logits.unsqueeze(1), size=(224,224), mode='bilinear', align_corners=False).squeeze(1)            

        
        return resized_logits, label

 # Define a function to process the dataset and save it
def process_and_save_dataset(words_segments, output_file):
    """
    Processes all items in the dataset and saves the processed objects and labels to a .pkl file.

    Parameters:
    - dataset: The dataset to process.
    - output_file: The path to the output .pkl file.
    """
    processed_data = []

    mfcc_dim={
        "n_mfcc":112, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    # Create dataset 
    segments_test = AudioSegmentDataset(words_segments, mfcc_dim, augment= False)

    print("Processing dataset...")
    for idx in tqdm(range(len(segments_test))):  # Use tqdm for progress bar
        try:
            processed_object, label = segments_test[idx]
            processed_data.append((processed_object.numpy(), label))  # Save as a tuple
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
  

    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print("Processing and saving complete!")  


def split_list_after_speaker(words_segments):
    """
    Groups words to their corresponding speakers and creates train test val split
    Returns:
    Train test val split with speakers
    """
    # Group word segments by speaker
    speaker_to_segments = defaultdict(list)
    for segment in words_segments:
        normalized_path = segment.path.replace("\\", "/")
        #print(normalized_path)
        _, filename = os.path.split(normalized_path)
        #print(filename)
        speaker = filename.replace('_sig', '')
        #print(speaker)
        speaker_to_segments[speaker].append(segment)
    # Get a list of unique speakers
    speakers = list(speaker_to_segments.keys())
    print("number speakers: ",np.shape(speakers))
    # Split speakers into training and testing sets
    speakers_train, speakers_test = train_test_split(speakers, random_state=42, test_size=0.05)
    speakers_train, speakers_val = train_test_split(speakers_train, random_state=42, test_size=0.15)

    # Collect word segments for each split
    segments_train = []
    segments_test = []
    segments_val = []
    print(f"Number of speakers in train: {len(speakers_train)}, val: {len(speakers_val)} test: {len(speakers_test)}")

    for speaker in speakers_train:
        segments_train.extend(speaker_to_segments[speaker])
    for speaker in speakers_val:
        segments_val.extend(speaker_to_segments[speaker])
    for speaker in speakers_test:
        segments_test.extend(speaker_to_segments[speaker])

    return segments_train, segments_val, segments_test


def find_pairs(audio_segments,phones_segments,index):
    """
    Takes a word which can be choosen by indices and searches for the correspüonding word in sig or normal. 
    Can also find all corresponding phones for a word.
    """
    sigmatism = None
    normal = None
    phones =["z","s","Z","S","ts"]
    phones_list_normal = []
    phones_list_sigmatism = []
    segment = audio_segments[index]###choose word here
    
    if segment.label_path == "sigmatism":
        print("It is Sigmatism")
        sigmatism = segment
        #get path from other file with normal speech
        matching_path = segment.path.replace("sigmatism", "normal")
        base, ext = os.path.splitext(matching_path)
        path = f"{base[:-4]}{ext}"
        print("PATH:",path)
        for normal in audio_segments:
            if (normal.label_path == "normal" and
                normal.label == segment.label and
                normal.path == path):
                print("Found normal pair")

                if(phones_segments):
                    for phone in phones_segments:
                        if (phone.label_path == "normal" and
                            phone.label in phones and
                            phone.path == path and
                            phone.sample_rate == sigmatism.label):
                            phones_list_normal.append(phone)

                        if (phone.label_path == "sigmatism" and
                            phone.label in phones and
                            phone.path == segment.path and
                            phone.sample_rate == sigmatism.label):
                            phones_list_sigmatism.append(phone)
                    return sigmatism, normal, phones_list_normal, phones_list_sigmatism
                return sigmatism, normal, phones_list_normal, phones_list_sigmatism

    
    if segment.label_path == "normal":
        print("It is Normal")
        normal =segment
        matching_path = segment.path.replace("normal", "sigmatism")
        base, ext = os.path.splitext(matching_path)
        path = f"{base}_sig{ext}"
        for sigmatism in audio_segments:
            if (sigmatism.label_path == "sigmatism" and
                sigmatism.label == normal.label and
                sigmatism.path == path):
                print("Found sigmatism pair")
                if(phones_segments):
                    for normal_phone in phones_segments:
                        if (normal_phone.label_path == "normal" and
                            normal_phone.label in phones and
                            normal_phone.path == normal.path and
                            normal_phone.sample_rate == sigmatism.label):
                            phones_list_normal.append(normal_phone)

                        if (normal_phone.label_path == "sigmatism" and
                            normal_phone.label in phones and
                            normal_phone.path == path and
                            normal_phone.sample_rate == sigmatism.label):
                            phones_list_sigmatism.append(normal_phone)
                    return sigmatism, normal, phones_list_normal, phones_list_sigmatism 
                return sigmatism, normal, phones_list_normal, phones_list_sigmatism 


if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    
    # words_segments = loader.create_dataclass_words()
    # loader.save_segments_to_pickle(words_segments, "words_segments.pkl")
    words_segments = loader.load_segments_from_pickle("words_without_normalization_for_labeling.pkl")
    phones_segments = loader.load_segments_from_pickle("data_lists/phones__24kHz.pkl")

    print(len(words_segments))
    mfcc_dim={
        "n_mfcc":112, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }




    segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)
    # segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)
    # train_samples = []
    # for f in segments_train:
    #     train_samples.append(f.audio_data)
    # print(np.shape(train_samples))
    # train_samples = np.concatenate(train_samples)
    # train_mean = np.mean(train_samples)
    # train_std = np.std(train_samples)
    # print("train_mean:", train_mean, " train_std:", train_std)
    for i in range(10):
        sigmatism, normal, phones_list_normal, phones_list_sigmatism = find_pairs(segments_test,phones_segments,i*30)
        #print(np.shape(phones_list_normal),np.shape(phones_list_sigmatism),sigmatism.label) 
        plotting.plot_mel_spectrogram(sigmatism)
        plotting.plot_mel_spectrogram(normal)
    # Create dataset 
    output_file = "STT_list_Interpolate_2D_train.pkl"  
    process_and_save_dataset(segments_train, output_file)
    output_file = "STT_list_Interpolate_2D_val.pkl"  
    process_and_save_dataset(segments_val, output_file)
    output_file = "STT_list_Interpolate_2D_test.pkl"  
    process_and_save_dataset(segments_test, output_file)


    segments_test = AudioSegmentDataset(words_segments, mfcc_dim, augment= False)
    logits,label = segments_test[0]  
    resized_array = logits.squeeze().detach().numpy()

    # Plot the resized logits as an image
    plt.figure(figsize=(10, 6))
    plt.imshow(resized_array, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Logit Intensity')
    plt.title("Resized Logits Visualization")
    plt.xlabel("Feature Dimension (Vocab Size)")
    plt.ylabel("Time Steps")
    plt.tight_layout()
    plt.savefig("LOGITS_INTERPOLATED.png")

    plt.show()

    