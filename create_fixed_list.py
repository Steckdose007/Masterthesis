import torch
from audiodataloader import AudioDataLoader, AudioSegment,split_list_after_speaker
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
import librosa
from dataclasses import dataclass
import data_augmentation
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.preprocessing import MinMaxScaler
import cv2




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
    """
    creates the list for training
    for every word stt heatmap, mel and mfcc are created
    modifies every word 3 times and puts it in the list 3 times
    """
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
    output_file = "mother_list_augment.pkl"
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
            augmented = True)
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
            augmented = True)
        data.append(featur_object)
        data.append(featur_object_noise)
        data.append(featur_object_pitch)

    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

import pickle
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler

# Function to augment a list of audio word segments
def augment_audio_list(word_segments):
    output_file = "augmented_audios_list.pkl"
    data = []

    # Iterate through each word segment
    for entry in tqdm(word_segments):
        # Extract relevant attributes from the entry
        start_time = entry.start_time
        end_time = entry.end_time
        audio_normal = entry.audio_data
        sample_rate = entry.sample_rate
        label_word = entry.label
        label_path = entry.label_path
        path = entry.path

        # Apply data augmentation: add noise and pitch shift
        audio_noise = data_augmentation.add_gaussian_noise(audio_normal, sample_rate)
        audio_pitch = data_augmentation.pitch_shift(audio_normal, sample_rate)

        # Create original TrainSegment (not augmented)
        featur_object = TrainSegment(
            start_time=start_time,
            end_time=end_time,
            data=audio_normal,
            sample_rate=sample_rate,
            label_word=label_word,
            label_path=label_path,
            path=path,
            augmented=False
        )

        # Create augmented segment with noise
        featur_object_noise = TrainSegment(
            start_time=start_time,
            end_time=end_time,
            data=audio_noise,
            sample_rate=sample_rate,
            label_word=label_word,
            label_path=label_path,
            path=path,
            augmented=True
        )

        # Create augmented segment with pitch shift
        featur_object_pitch = TrainSegment(
            start_time=start_time,
            end_time=end_time,
            data=audio_pitch,
            sample_rate=sample_rate,
            label_word=label_word,
            label_path=label_path,
            path=path,
            augmented=True
        )

        # Append original and augmented segments to the data list
        data.append(featur_object)
        data.append(featur_object_noise)
        data.append(featur_object_pitch)

    # Save all augmented data to a pickle file
    print(f"Saving processed data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

# Function to compute mean and standard deviation of raw audio data for normalization
def compute_mean_sdt_for_normalization_audio(data):
    # Split data into training, validation, and test sets
    segments_train, segments_val, segments_test = split_list_after_speaker(data)

    train_samples = [f.audio_data for f in segments_train]
    print(np.shape(train_samples))

    # Concatenate all training samples into a single array
    train_samples = np.concatenate(train_samples)

    # Compute mean and standard deviation of the training audio
    train_mean = np.mean(train_samples)
    train_std = np.std(train_samples)

    print("train_mean:", train_mean, " train_std:", train_std)

# Function to compute normalization statistics (mean, std) for MFCC or Mel features
def compute_mean_std_for_mfccormel_normalization(words_list):
    # Split data into training, validation, and test sets
    segments_train, segments_val, segments_test = split_list_after_speaker(words_list)

    # Resize all mel features to 224x224 and collect them
    resized_mfccs = [
        cv2.resize(segment.mel, (224, 224), interpolation=cv2.INTER_LINEAR)
        for segment in segments_train
    ]

    # Concatenate all frames for global statistics computation
    all_mfcc_frames = np.concatenate(resized_mfccs)

    # Normalize using Min-Max scaling to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_mfcc_scaled = scaler.fit_transform(all_mfcc_frames)

    # Compute global mean and standard deviation
    global_mean = np.mean(all_mfcc_scaled)
    global_std = np.std(all_mfcc_scaled)

    print("global_mean  :  ", global_mean)
    print("global_std   :  ", global_std)

# Function to compute normalization statistics for STT (e.g., spectrogram) embeddings
def compute_mean_std_for_stt_normalization(words_list):
    # Split data into training, validation, and test sets
    segments_train, segments_val, segments_test = split_list_after_speaker(words_list)

    # Resize all STT features to 224x224 and collect them
    resized_mfccs = [
        cv2.resize(segment.stt.detach().cpu().numpy()[0], (224, 224), interpolation=cv2.INTER_LINEAR)
        for segment in segments_train
    ]

    # Concatenate all frames for global statistics computation
    all_mfcc_frames = np.concatenate(resized_mfccs)

    # Normalize using Min-Max scaling to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_mfcc_scaled = scaler.fit_transform(all_mfcc_frames)

    # Compute global mean and standard deviation
    global_mean = np.mean(all_mfcc_scaled)
    global_std = np.std(all_mfcc_scaled)

    print("global_mean  :  ", global_mean)
    print("global_std   :  ", global_std)


if __name__ == "__main__":
    #============================create fixedlists ==========================================
    # Before change what you want to have in the dataloader
    #for train put every word in there 3 times
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    #phones_segments = loader.load_segments_from_pickle("data_lists\phone_normalized_16kHz.pkl")
    words_segments = loader.load_segments_from_pickle("data_lists/words_normalized_16kHz.pkl")
    print(np.shape(words_segments))
    #word = words_segments[10000]
    #print(np.shape(word.mfcc),np.shape(word.mel),np.shape(word.stt.detach().cpu().numpy()[0]))
    #compute_mean_std_for_stt_normalization(words_segments)
    #word = words_segments[100]
    #plotting.visualize_augmentations(word.audio_data,word.sample_rate)
    #augment_audio_list(words_segments[:60])
    create_list(words_segments)
