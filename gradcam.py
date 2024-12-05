import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D , CNNMFCC,initialize_mobilenet
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset , plot_mfcc
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from collections import defaultdict
import librosa
from torch.nn.functional import interpolate
from torchsummary import summary

def split_list_after_speaker(words_segments):
    # Group word segments by speaker
    speaker_to_segments = defaultdict(list)
    for segment in words_segments:
        speaker = os.path.basename(segment.path).replace('_sig', '')
        speaker_to_segments[speaker].append(segment)
    # Get a list of unique speakers
    speakers = list(speaker_to_segments.keys())
    print("number speakers: ",np.shape(speakers))
    # Split speakers into training and testing sets
    speakers_train, speakers_test = train_test_split(speakers, random_state=42, test_size=0.20)
    
    # Collect word segments for each split
    segments_train = []
    segments_test = []
    print(f"Number of speakers in train: {len(speakers_train)}, test: {len(speakers_test)}")

    for speaker in speakers_train:
        segments_train.extend(speaker_to_segments[speaker])

    for speaker in speakers_test:
        segments_test.extend(speaker_to_segments[speaker])
    return segments_train, segments_test

def grad_cam_heatmap(model, input_tensor, target_class):
    """
    Generates a Grad-CAM heatmap for the given input and target class.
    """
    # Ensure input tensor has the right dimensions
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    input_tensor = input_tensor.to(next(model.parameters()).device)  # Move to the correct device
    model.eval()

    # Identify the last convolutional layer for Grad-CAM
    target_layer = model.features[-1]  # Use the last layer of features
    
    gradients = []
    activations = []

    # Register hooks to capture gradients and activations
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    _, predicted_class = torch.max(output, dim=1)
    loss = output[:, target_class].sum()  # Target class activation
    print("Label: ",target_class," Prediction: ",predicted_class.numpy())
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()

    # Extract the gradients and activations
    gradients = gradients[0].detach().numpy()
    activations = activations[0].detach().numpy()

    # Compute the weights for Grad-CAM
    weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
    cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)  # Weighted sum of activations

    # ReLU on the heatmap to remove negative values
    cam = np.maximum(cam, 0)

    # Normalize the heatmap to [0, 1]
    cam = cam[0]  # Remove batch dimension
    cam = cam / np.max(cam)
    # Resize the heatmap to match the input size
    cam_resized = interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                                size=(input_tensor.shape[2], input_tensor.shape[3]),  # Match input height and width
                                mode='bilinear',
                                align_corners=False).squeeze().numpy()
    return cam_resized

def overlay_heatmap_with_input(input_tensor, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap onto the input tensor (e.g., mel-spectrogram).
    
    Parameters:
    - input_tensor: The original input (e.g., mel-spectrogram), shape (n_mfcc, time_frames)
    - heatmap: The Grad-CAM heatmap, shape (n_mfcc, time_frames)
    - alpha: Transparency of the overlay (0: fully input, 1: fully heatmap)
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()
    heatmap = heatmap.squeeze()
    input_tensor = input_tensor.squeeze()
    print(np.shape(heatmap))
    print(np.shape(input_tensor))

    # Normalize the input tensor to [0, 1] for visualization
    input_tensor = input_tensor - np.min(input_tensor)
    input_tensor /= np.max(input_tensor) if np.max(input_tensor) != 0 else 1

    # Create the overlay
    overlay = alpha * heatmap + (1 - alpha) * input_tensor

    # Plot the overlay
    plt.figure(figsize=(12, 6))
    plt.imshow(overlay, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Heatmap Intensity')
    plt.title("Overlay of Heatmap and Input Tensor")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.grid(False)
    plt.show()
    # Plot the overlay
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Heatmap Intensity')
    plt.title("heatmap")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.grid(False)
    plt.show()

def overlay_heatmap_on_mel_spectrogram(signal, sample_rate, model, input_tensor, target_class):
    """
    Generate a Mel-spectrogram and overlay Grad-CAM heatmap.

    Args:
    - signal: Audio signal.
    - sample_rate: Sample rate of the signal.
    - model: Trained PyTorch model.
    - input_tensor: Input tensor of shape [1, 1, n_mfcc, time_frames].
    - target_class: Target class index.

    Returns:
    - None (plots the visualization).
    """
    # Generate Grad-CAM heatmap
    heatmap = grad_cam_heatmap(model, input_tensor, target_class)
    overlay_heatmap_with_input(input_tensor,heatmap)
    # Rescale heatmap to Mel-spectrogram dimensions
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.005 * sample_rate) 
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128,n_fft=2048,hop_length=hop_length,
                                                     win_length=frame_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Create the overlay
    overlay = 0.2 * heatmap + (1 - 0.2) * mel_spectrogram_db

    # Plot the overlay
    plt.figure(figsize=(12, 6))
    plt.imshow(overlay, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Heatmap Intensity')
    plt.title("Overlay of Heatmap and Input Tensor")
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC Coefficients")
    plt.grid(False)
    plt.show()

def explain_model(dataset):
    #path = "models/Mobilenet_20241129-170942.pth"
    path = "models/Mobilenet_20241204-080257.pth"
    model = initialize_mobilenet(num_classes=2, input_channels=1)
    model.load_state_dict(torch.load(path, weights_only=True))
    #model.to(device)
    #summary(model, input_size=(1, 224, 224))
    input_tensor , target_class,signal = dataset[229]#select mfcc
    plot_mfcc(input_tensor)
    overlay_heatmap_on_mel_spectrogram(signal, 24000, model, input_tensor, target_class)


if __name__ == "__main__":
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    segments_train, segments_test = split_list_after_speaker(words_segments)
    print(f"Number of word segments in train: {len(segments_train)}, test: {len(segments_test)}")

    mfcc_dim={
        "n_mfcc":112, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    # Create dataset 
    segments_test = AudioSegmentDataset(segments_test, mfcc_dim, augment= False)
    segments_train = AudioSegmentDataset(segments_train, mfcc_dim, augment = True)
    explain_model(segments_test)
    