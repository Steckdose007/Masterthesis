import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D , CNNMFCC,initialize_mobilenet,initialize_mobilenetV3
from audiodataloader import AudioDataLoader, AudioSegment, split_list_after_speaker
from Dataloader_gradcam import GradcamDataset , plot_mfcc,TrainSegment
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from collections import defaultdict
import librosa
from torch.nn.functional import interpolate
import pickle


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
    return cam_resized,target_class,predicted_class.numpy()

def overlay_heatmap_with_input(input_tensor, heatmap, word_spoken, label, predicted, alpha=0.6):
    """
    Overlay the Grad-CAM heatmap onto the input tensor (e.g., mel-spectrogram) 
    with improved visibility by using distinct colormaps.
    """
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()
    
    heatmap = heatmap.squeeze()
    input_tensor = input_tensor.squeeze()

    # Normalize the input tensor and heatmap to [0, 1]
    input_tensor = normalize_image(input_tensor)
    heatmap = normalize_image(heatmap)

    # Create the overlay by first plotting the input in grayscale
    # and then overlaying the heatmap with a vibrant colormap (e.g., 'jet').
    overlay = alpha * heatmap + (1 - alpha) * input_tensor

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Input Tensor (use grayscale to provide contrast)
    im0 = axs[0, 0].imshow(input_tensor, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 0].set_title(f"Input Tensor for word {word_spoken}")
    axs[0, 0].set_xlabel("Time Frames")
    axs[0, 0].set_ylabel("Mel Coefficients")
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot Overlay (input in grayscale, heatmap in 'jet')
    axs[0, 1].imshow(input_tensor, aspect='auto', origin='lower', cmap='gray')
    im1 = axs[0, 1].imshow(heatmap, aspect='auto', origin='lower', cmap='inferno', alpha=alpha)
    axs[0, 1].set_title(f"Overlay of Heatmap and Input Tensor for word {word_spoken}")
    axs[0, 1].set_xlabel("Time Frames")
    axs[0, 1].set_ylabel("Mel Coefficients")
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot Heatmap alone
    im2 = axs[1, 0].imshow(heatmap, aspect='auto', origin='lower', cmap='inferno')
    axs[1, 0].set_title("Heatmap")
    axs[1, 0].set_xlabel("Time Frames")
    axs[1, 0].set_ylabel("Mel Coefficients")
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Optionally, plot the combined overlay again in a separate plot
    im3 = axs[1, 1].imshow(overlay, aspect='auto', origin='lower', cmap='inferno')
    axs[1, 1].set_title(f"Alpha Blended Overlay Label: {label} Prediction: {predicted}")
    axs[1, 1].set_xlabel("Time Frames")
    axs[1, 1].set_ylabel("Mel Coefficients")
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.show()

def overlay_heatmap_with_input_2channel(input_tensor, heatmap, word_spoken, label, predicted, alpha=0.6):
    """
    Overlay the Grad-CAM heatmap onto the input tensor (e.g., mel-spectrogram) 
    with improved visibility by using distinct colormaps.
    """
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()
    
    heatmap = heatmap.squeeze()
    input_tensor = input_tensor.squeeze()

    mel = normalize_image(input_tensor[0])
    att = normalize_image(input_tensor[1])
    # Normalize the input tensor and heatmap to [0, 1]
    heatmap = normalize_image(heatmap)

    # Create the overlay by first plotting the input in grayscale
    # and then overlaying the heatmap with a vibrant colormap (e.g., 'jet').
    overlay = alpha * heatmap + (1 - alpha) * input_tensor

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Input Tensor (use grayscale to provide contrast)
    im0 = axs[0, 0].imshow(mel, aspect='auto', origin='lower', cmap='inferno')
    axs[0, 0].set_title(f"Input Tensor for word {word_spoken}")
    axs[0, 0].set_xlabel("Time Frames")
    axs[0, 0].set_ylabel("Mel Coefficients")
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot Overlay (input in grayscale, heatmap in 'jet')
    axs[0, 1].imshow(mel, aspect='auto', origin='lower', cmap='gray')
    im1 = axs[0, 1].imshow(heatmap, aspect='auto', origin='lower', cmap='inferno', alpha=alpha)
    axs[0, 1].set_title(f"Overlay of Heatmap and Input Tensor for word {word_spoken}")
    axs[0, 1].set_xlabel("Time Frames")
    axs[0, 1].set_ylabel("Mel Coefficients")
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot Heatmap alone
    im2 = axs[1, 0].imshow(att, aspect='auto', origin='lower', cmap='inferno')
    axs[1, 0].set_title("Heatmap")
    axs[1, 0].set_xlabel("Time Frames")
    axs[1, 0].set_ylabel("Mel Coefficients")
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Optionally, plot the combined overlay again in a separate plot
    axs[1, 1].imshow(att, aspect='auto', origin='lower', cmap='gray')
    im11 = axs[1, 1].imshow(heatmap, aspect='auto', origin='lower', cmap='inferno', alpha=alpha)
    axs[1, 1].set_title(f"Alpha Blended Overlay Label: {label} Prediction: {predicted}")
    axs[1, 1].set_xlabel("Time Frames")
    axs[1, 1].set_ylabel("Mel Coefficients")
    plt.colorbar(im11, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.show()

def explain_model(dataset):
    #path = "models/Mobilenet_20241129-170942.pth"
    #'path to best model'
    path = "models\MEL+ATT_ohneschedluer_Train+val20250210-190753.pth"
    model = initialize_mobilenetV3(num_classes=2, dropout = 0.3, input_channels=2)
    model.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device('cpu')))
    #model.to(device)
    #summary(model, input_size=(1, 224, 224))
    input_tensor, target_class, word_spoken = find_spezific_word(dataset,"Satz",0)
    heatmap,target_class,predicted_class = grad_cam_heatmap(model, input_tensor, target_class)
    overlay_heatmap_with_input_2channel(input_tensor,heatmap, word_spoken,target_class,predicted_class)
    #input_tensor , target_class,word_spoken = dataset[232]#select 
    #heatmap,target_class,predicted_class = grad_cam_heatmap(model, input_tensor, target_class)
    #overlay_heatmap_with_input(input_tensor,heatmap, word_spoken,target_class,predicted_class)

def find_spezific_word(dataset, word, label):
    for i in range(len(dataset)):
        input_tensor, target_class, word_spoken = dataset[i+100]#adjust the 100 because it finds only the next index
        if(word_spoken == word and label == target_class):
            print(i)
            return input_tensor, target_class, word_spoken
    print("No word Found.....................................!")
    return None



if __name__ == "__main__":
    with open("data_lists\mother_list_augment.pkl", "rb") as f:
        data = pickle.load(f)
    segments_train, segments_val, segments_test= split_list_after_speaker(data)
    print(f"Number of word segments in train: {len(segments_train)}, test: {len(segments_test)}")
    # Create dataset 
    segments_test = GradcamDataset(segments_test)
    segments_train = GradcamDataset(segments_train)
    #Use test set to see how good model looks
    explain_model(segments_test)
    