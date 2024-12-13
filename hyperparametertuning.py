import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import CNNMFCC  , initialize_mobilenet
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset 
import optuna
from optuna.visualization import plot_param_importances,plot_param_importances
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
# Global dataset variable
global segments_train, segments_val, phones_segments
# Load the dataset once
def split_list_after_speaker(words_segments):
    # Group word segments by speaker
    speaker_to_segments = defaultdict(list)
    for segment in words_segments:
        speaker = os.path.basename(segment.path).replace('_sig', '')
        speaker_to_segments[speaker].append(segment)
    # Get a list of unique speakers
    speakers = list(speaker_to_segments.keys())
    #print("number speakers: ",np.shape(speakers))
    # Split speakers into training and testing sets
    speakers_train, speakers_test = train_test_split(speakers, random_state=42, test_size=0.05)
    speakers_train, speakers_val = train_test_split(speakers_train, random_state=42, test_size=0.15)

    # Collect word segments for each split
    segments_train = []
    segments_test = []
    segments_val = []
    #print(f"Number of speakers in train: {len(speakers_train)}, val: {len(speakers_val)} test: {len(speakers_test)}")

    for speaker in speakers_train:
        segments_train.extend(speaker_to_segments[speaker])
    for speaker in speakers_val:
        segments_val.extend(speaker_to_segments[speaker])
    for speaker in speakers_test:
        segments_test.extend(speaker_to_segments[speaker])

    return segments_train, segments_val, segments_test

def prepare_dataset():
    global segments_train, segments_val, phones_segments
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    phones_segments = loader.load_segments_from_pickle("phones_atleast2048long_24kHz.pkl")
    segments_train, segments_val, segments_test = split_list_after_speaker(words_segments)
    #print(f"Number of word segments in train: {len(segments_train)}, test: {len(segments_val)}")
    
    return segments_train,segments_val

# Define objective function for Optuna
def objective(trial):
    global segments_train, segments_val, phones_segments

    # Define search space
    gamma = trial.suggest_float("gamma", 0.5, 0.99)
    step_size = trial.suggest_int("step_size", 5, 50)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Use log=True for logarithmic scale
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mfcc_dim={
        "n_mfcc":128, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    # Create dataset 
    segments_val = AudioSegmentDataset(segments_val, phones_segments, mfcc_dim, augment= False)
    segments_train = AudioSegmentDataset(segments_train, phones_segments, mfcc_dim, augment = True)
    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(segments_val, batch_size=batch_size, shuffle=False)
   
    # Define model
    num_classes = 2  # Change as needed
    input_channels = 1
    model = initialize_mobilenet(num_classes, input_channels).to(device)    
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Training loop
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Step the scheduler at the end of each epoch
        scheduler.step()
        #     # Track accuracy and loss
        #     _, predicted = torch.max(outputs.data, 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels).sum().item()
        #     running_loss += loss.item()

        # train_loss = running_loss / len(train_loader)
        # train_accuracy = correct / total

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(val_loader)
    test_accuracy = correct / total

    # Report test accuracy as the objective to optimize
    return test_accuracy

def optimize_with_progress(study, objective, n_trials):
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])


if __name__ == "__main__":
    segments_train, segments_test = prepare_dataset()
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    # Optimize 
    optimize_with_progress(study, objective, n_trials=50)    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters
    with open("best_hyperparameters_Approach1.txt", "w") as f:
        f.write(f"Best trial accuracy: {trial.value}\n")
        f.write("Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")

    fig = plot_param_importances(study)
    fig.show()
    param_importance_plot = plot_param_importances(study)
    param_importance_plot.show()