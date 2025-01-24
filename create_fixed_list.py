import torch
from audiodataloader import AudioDataLoader, AudioSegment
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
import librosa
from dataclasses import dataclass
import data_augmentation
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor





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

def compute_mfcc_features(signal, sample_rate, n_mfcc=112, n_mels=128, frame_size=25e-3, hop_size=5e-3, n_fft=2048):
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

def compute_melspectogram_features(signal, sample_rate=16000, n_mels=128, frame_size=25e-3, hop_size=5e-3, n_fft=2048):
        # Convert frame and hop size from seconds to samples
        frame_length = int(frame_size * sample_rate)
        hop_length = int(hop_size * sample_rate)   
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,win_length=frame_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db

def make_STT_heatmap(audio,processor,model):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits  = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    return logits


def create_list(word_segments):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    feature_dim={
        "n_mfcc":112, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    output_file = "mother_list.pkl"
    data = []
    for entry in tqdm(word_segments):
        start_time = entry.start_time
        end_time = entry.end_time
        audio_normal = entry.audio_data
        sample_rate = entry.sample_rate
        label_word = entry.label
        label_path = entry.label_path
        path = entry.path

        #Put every word 3 times in the list. Normal , with noise, with pitch
        audio_noise = data_augmentation.add_gaussian_noise(audio_normal,sample_rate)
        audio_pitch = data_augmentation.pitch_shift(audio_normal,sample_rate)
        """mel specto"""
        feature_normal_mel = compute_melspectogram_features(audio_normal,sample_rate,feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        feature_noise_mel = compute_melspectogram_features(audio_noise,sample_rate,feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        feature_pitch_mel = compute_melspectogram_features(audio_pitch,sample_rate,feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        """mfcc"""
        feature_normal_mfcc = compute_mfcc_features(audio_normal,sample_rate,feature_dim["n_mfcc"],feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        feature_noise_mfcc = compute_mfcc_features(audio_noise,sample_rate,feature_dim["n_mfcc"],feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        feature_pitch_mfcc = compute_mfcc_features(audio_pitch,sample_rate,feature_dim["n_mfcc"],feature_dim["n_mels"],feature_dim["frame_size"],feature_dim["hop_size"],feature_dim["n_fft"])
        """STT"""
        feature_normal_stt = make_STT_heatmap(audio_normal,processor,model)
        feature_noise_stt = make_STT_heatmap(audio_noise,processor,model)
        feature_pitch_stt = make_STT_heatmap(audio_pitch,processor,model)

        featur_object = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            mfcc = feature_normal_mfcc,
            mel = feature_normal_mel,
            stt = feature_normal_stt,
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        featur_object_noise = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            mfcc = feature_noise_mfcc,
            mel = feature_noise_mel,
            stt = feature_noise_stt,            
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        featur_object_pitch = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            mfcc = feature_pitch_mfcc,
            mel = feature_pitch_mel,
            stt = feature_pitch_stt,            
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        data.append(featur_object)
        data.append(featur_object_noise)
        data.append(featur_object_pitch)

    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def augment_audio_list(word_segments):
    
    
    output_file = "augmented_audios_list.pkl"
    data = []
    for entry in tqdm(word_segments):
        start_time = entry.start_time
        end_time = entry.end_time
        audio_normal = entry.audio_data
        sample_rate = entry.sample_rate
        label_word = entry.label
        label_path = entry.label_path
        path = entry.path

        #Put every word 3 times in the list. Normal , with noise, with pitch
        audio_noise = data_augmentation.add_gaussian_noise(audio_normal,sample_rate)
        audio_pitch = data_augmentation.pitch_shift(audio_normal,sample_rate)
        featur_object = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            data = audio_normal,
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        featur_object_noise = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            data = audio_noise,
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        featur_object_pitch = TrainSegment(
            start_time = start_time,
            end_time = end_time,
            data = audio_pitch,
            sample_rate = sample_rate,
            label_word = label_word,
            label_path = label_path,
            path = path,
            augmented = False)
        data.append(featur_object)
        data.append(featur_object_noise)
        data.append(featur_object_pitch)

    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    #============================create fixedlists ==========================================
    # Before change what you want to have in the dataloader
    #for train put every word in there 3 times
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    #phones_segments = loader.load_segments_from_pickle("data_lists\phone_normalized_16kHz.pkl")
    words_segments = loader.load_segments_from_pickle("data_lists/words_normalized_16kHz.pkl")
    #word = words_segments[100]
    #plotting.visualize_augmentations(word.audio_data,word.sample_rate)
    #augment_audio_list(words_segments[:60])
    create_list(words_segments)
