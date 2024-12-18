import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D , CNNMFCC,initialize_mobilenet
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset 
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

def train_model(model, train_loader, test_loader, criterion, optimizer,scheduler, num_epochs=10,best_model_filename = None):
    best_loss = 1000000  # To keep track of the best accuracy
    train_losses = []
    test_losses = []
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar using tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Update the progress bar with average loss
            progress_bar.set_postfix({'Loss': running_loss / total , 'Accuracy': correct / total})

        # Calculate epoch loss and accuracy for training data
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        # Step the scheduler
        #scheduler.step()
        # Evaluate on the test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        test_losses.append(test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            #best_model_filename = os.path.join('models', best_model_filename)
            torch.save(model.state_dict(), os.path.join('models', best_model_filename))
            print(f"Best model saved with test accuracy {best_loss:.4f} as {best_model_filename}")
    
    # Plot train and test losses
    plot_losses(train_losses, test_losses,best_model_filename,best_test_acc)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0  # To accumulate the total number of samples
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate loss and accuracy
            batch_size = inputs.size(0)  # Get the batch size
            running_loss += loss.item() * batch_size  # Multiply by batch size to account for smaller last batch
            total_samples += batch_size
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Compute average loss per sample
    average_loss = running_loss / total_samples  # Divide by total number of samples
    accuracy = correct / total  # Total accuracy

    return average_loss, accuracy
# Function to plot training and test loss
def plot_losses(train_losses, test_losses,best_model_filename,best_test_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss over Epochs with Acc of: "+str(best_test_acc))
    plt.legend()
    plt.grid(True)
    plt.savefig('models/loss_plot'+best_model_filename+'.png')  # Save the plot as an image
    plt.show()

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


if __name__ == "__main__":
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    phones_segments = loader.load_segments_from_pickle("phones_atleast2048long_24kHz.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)

    print(f"Number of word segments in train: {len(segments_train)},val: {len(segments_val)} test: {len(segments_test)}")
    # Set target length for padding/truncation
    # maximum word lenght is 65108 and because a strechtching of up to 120% can appear the buffer hast to be that big.
    target_length_24kHz_MFCC = int(224)#data augmentstion already done and number of frames.
    target_length = target_length_24kHz_MFCC

    # Hyperparameters
    n_mfcc = 112 # Number of MFCC coefficients
    num_classes = 2  # Adjust based on your classification task (e.g., binary classification for sigmatism)
    learning_rate = 0.00001
    num_epochs = 50
    batch_size = 16

    mfcc_dim={
        "n_mfcc":n_mfcc, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    # Create dataset 
    segments_val = AudioSegmentDataset(segments_val,phones_segments, mfcc_dim, augment= False)
    segments_train = AudioSegmentDataset(segments_train,phones_segments, mfcc_dim, augment = True)
    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(segments_val, batch_size=batch_size, shuffle=False)


    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
    num_classes = 2  # Change as needed
    input_channels = 1  # Since your input is grayscale spectrogram
    model = initialize_mobilenet(num_classes, input_channels).to(device)
    #model = CNNMFCC(num_classes, n_mfcc,target_length).to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=24, gamma=0.7)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,eta_min=0.00001)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model_filename = f"Mobilenet_Attention_Approach2_{timestamp}.pth"
    
    train_model(model, train_loader, val_loader, criterion, optimizer,None, num_epochs=num_epochs,best_model_filename=best_model_filename)
