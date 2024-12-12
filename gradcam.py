import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D , CNNMFCC,initialize_mobilenet
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_gradcam import GradcamDataset , plot_mfcc
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

def normalize_image(image):
    """
    Normalize an image so that its values are between 0 and 1.

    Parameters:
    - image: The input image as a 2D or 3D numpy array.

    Returns:
    - Normalized image with values in the range [0, 1].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Avoid division by zero if the image is constant
    if max_val - min_val == 0:
        return np.zeros_like(image)
    
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


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

def overlay_heatmap_with_input(input_tensor, heatmap, padding, mel_tensor,padding_mel ,alpha=0.4):
    """
    Overlay the Grad-CAM heatmap onto the input tensor (e.g., mel-spectrogram).
    
    Parameters:
    - input_tensor: The original input (e.g., mel-spectrogram), shape (n_mfcc, time_frames)
    - heatmap: The Grad-CAM heatmap, shape (n_mfcc, time_frames)
    - alpha: Transparency of the overlay (0: fully input, 1: fully heatmap)
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()
    if isinstance(mel_tensor, torch.Tensor):
        mel_tensor = mel_tensor.cpu().numpy()
    heatmap = heatmap.squeeze()
    input_tensor = input_tensor.squeeze()
    mel_tensor = mel_tensor.squeeze()

    print(np.shape(heatmap))
    print(np.shape(input_tensor))
    print(np.shape(mel_tensor))

    heatmap = heatmap[:224-padding[0],:224-padding[1]]
    input_tensor = input_tensor[:224-padding[0],:224-padding[1]]
    mel_tensor = mel_tensor[:224-padding_mel[0],:224-padding_mel[1]]

    print(np.shape(heatmap))
    print(np.shape(input_tensor))
    print(np.shape(mel_tensor))

    # Normalize the input tensor to [0, 1] for visualization
    input_tensor = normalize_image(input_tensor)
    heatmap = normalize_image(heatmap)
    mel_tensor=normalize_image(mel_tensor)

    # Create the overlay
    overlay = alpha * heatmap + (1 - alpha) * input_tensor

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Input Tensor
    axs[0, 0].imshow(input_tensor, aspect='auto', origin='lower', cmap='plasma')
    axs[0, 0].set_title("Input Tensor")
    axs[0, 0].set_xlabel("Time Frames")
    axs[0, 0].set_ylabel("MFCC Coefficients")
    axs[0, 0].grid(False)
    axs[0, 0].colorbar = plt.colorbar(axs[0, 0].imshow(input_tensor, aspect='auto', origin='lower', cmap='plasma'),
                                       ax=axs[0, 0], format='%+2.0f')
    
    # Plot Overlay
    axs[0, 1].imshow(overlay, aspect='auto', origin='lower', cmap='plasma')
    axs[0, 1].set_title("Overlay of Heatmap and Input Tensor")
    axs[0, 1].set_xlabel("Time Frames")
    axs[0, 1].set_ylabel("MFCC Coefficients")
    axs[0, 1].grid(False)
    axs[0, 1].colorbar = plt.colorbar(axs[0, 1].imshow(overlay, aspect='auto', origin='lower', cmap='plasma'),
                                       ax=axs[0, 1], format='%+2.0f')
    
    # Plot Heatmap
    axs[1, 0].imshow(heatmap, aspect='auto', origin='lower', cmap='plasma')
    axs[1, 0].set_title("Heatmap")
    axs[1, 0].set_xlabel("Time Frames")
    axs[1, 0].set_ylabel("MFCC Coefficients")
    axs[1, 0].grid(False)
    axs[1, 0].colorbar = plt.colorbar(axs[1, 0].imshow(heatmap, aspect='auto', origin='lower', cmap='plasma'),
                                       ax=axs[1, 0], format='%+2.0f')

    # Plot Mel Spectrogram
    axs[1, 1].imshow(mel_tensor, aspect='auto', origin='lower', cmap='plasma')
    axs[1, 1].set_title("Mel Spectrogram")
    axs[1, 1].set_xlabel("Time Frames")
    axs[1, 1].set_ylabel("Mel Frequency (Hz)")
    axs[1, 1].grid(False)
    axs[1, 1].colorbar = plt.colorbar(axs[1, 1].imshow(mel_tensor, aspect='auto', origin='lower', cmap='plasma'),
                                       ax=axs[1, 1], format='%+2.0f dB')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def explain_model(dataset):
    #path = "models/Mobilenet_20241129-170942.pth"
    path = "models/Mobilenet_othernormalization_20241205-082953.pth"
    model = initialize_mobilenet(num_classes=2, input_channels=1)
    model.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device('cpu')))
    #model.to(device)
    #summary(model, input_size=(1, 224, 224))
    input_tensor , target_class,word_spoken,signal,padding, mel_tensor,padding_mel = find_spezific_word(dataset,"Niesen",0)
    heatmap = grad_cam_heatmap(model, input_tensor, target_class)
    overlay_heatmap_with_input(input_tensor,heatmap,padding, mel_tensor,padding_mel)
    input_tensor , target_class,word_spoken,signal,padding, mel_tensor,padding_mel = dataset[232]#select mfcc
    #plot_mfcc(input_tensor)
    heatmap = grad_cam_heatmap(model, input_tensor, target_class)
    overlay_heatmap_with_input(input_tensor,heatmap,padding, mel_tensor,padding_mel)

def find_spezific_word(dataset, word, label):
    for i in range(len(dataset)):
        input_tensor , target_class,word_spoken,signal,padding, mel_tensor,padding_mel = dataset[i+40]
        if(word_spoken == word and label == target_class):
            return input_tensor , target_class,word_spoken,signal,padding, mel_tensor,padding_mel
    print("No word Found.....................................!")
    return None



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
    segments_test = GradcamDataset(segments_test, mfcc_dim, augment= False)
    segments_train = GradcamDataset(segments_train, mfcc_dim, augment = True)
    explain_model(segments_test)
    