import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D , CNNMFCC,initialize_mobilenet,initialize_mobilenetV3
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset ,process_and_save_dataset
from sklearn.model_selection import train_test_split
import datetime
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from collections import defaultdict
import librosa
from Dataloader_fixedlist import FixedListDataset
from torch.nn.functional import interpolate


if __name__ == "__main__":
    #============================create fixedlists ==========================================
    # Before change what you want to have in the dataloader
    #for train put every word in there 3 times
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    phones_segments = loader.load_segments_from_pickle("phones__24kHz.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)
    
