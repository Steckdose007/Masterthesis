import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D ,modelSST,  CNNMFCC,initialize_mobilenet, initialize_mobilenetV3, initialize_mobilenetV3small
from audiodataloader import AudioDataLoader, AudioSegment, find_pairs, split_list_after_speaker
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import datetime
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from collections import defaultdict
import librosa
from Dataloader_fixedlist import FixedListDataset,FixedListDatasetvalidation
from create_fixed_list import TrainSegment
from torch.nn.functional import interpolate
#from torchsummary import summary
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def train_model(model, train_loader, test_loader, criterion, optimizer,scheduler, num_epochs=10,best_model_filename = None):
    best_loss = 1000000  # To keep track of the best accuracy
    train_losses = []
    val_losses = []
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
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_test_acc:
            best_loss = val_loss
            best_test_acc = val_acc
            #best_model_filename = os.path.join('models', best_model_filename)
            torch.save(model.state_dict(), os.path.join('models', best_model_filename))
            print(f"Best model saved with val accuracy {best_test_acc:.4f} as {best_model_filename}")
    
    # Plot train and test losses
    plot_losses(train_losses, val_losses,best_model_filename,best_test_acc)

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
    #plt.show()

def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    Computes several performance metrics.
    """
    RR = accuracy_score(y_true, y_pred)  # Overall accuracy
    Rn = recall_score(y_true, y_pred, pos_label=0)  # Recall for Normal class
    Rp = recall_score(y_true, y_pred, pos_label=1)  # Recall for Pathological class
    CL = (Rn + Rp) / 2   # Class-wise averaged recognition rate
    AUC = roc_auc_score(y_true, y_pred_proba[:, 1])  # AUC using probabilities for positive class
    return {
        'RR': float(round(RR, 3)),
        'Rn': float(round(Rn, 3)),
        'Rp': float(round(Rp, 3)),
        'CL': float(round(CL, 3)),
        'AUC': float(round(AUC, 3))
    }

def evaluate_model_metrics(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_pred_probas = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Get probabilities from the outputs
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_pred_probas.extend(probabilities.cpu().numpy())
    
    avg_loss = running_loss / len(test_loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_pred_probas))
    return avg_loss, metrics

if __name__ == "__main__":
    
    # loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Hyperparameters
    mfcc_dim={
        "n_mfcc":128, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    Hyperparameters={
        "gamma": 0.7301331136239125,
        "step_size": 24,
        "learning_rate": 0.0007155660444277831,
        "batch_size": 128,
        "momentum": 0.9393347731944004,
        "weight_decay":5.495804018652414e-05,
        "dropout":0.5
    }
    num_classes = 2  #  binary classification for sigmatism
    learning_rate = Hyperparameters["learning_rate"]
    num_epochs = 50
    batch_size = Hyperparameters["batch_size"]
    step_size = Hyperparameters["step_size"]
    gamma=Hyperparameters["gamma"]
    momentum=Hyperparameters["momentum"]
    dropout = Hyperparameters["dropout"]
    weight_decay = Hyperparameters["weight_decay"]
    #============================Load fixed lists =====================================
    with open("data_lists\mother_list_augment.pkl", "rb") as f:
        data = pickle.load(f)
    segments_train, segments_val, segments_test= split_list_after_speaker(data)
    combined_train_val = segments_train + segments_val
    segments_train = FixedListDataset(combined_train_val)
    segments_test = FixedListDatasetvalidation(segments_test)
    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True,num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True)  # Fetches 2x the batch size in advance)
    test_loader = DataLoader(segments_test, batch_size=batch_size, shuffle=False)#,num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True) 


    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
    num_classes = 2  # Change as needed
    input_channels = 2  #input is grayscale spectrogram
    model = initialize_mobilenetV3(num_classes,dropout, input_channels)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)  # Move model to GPU(s)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)# L2 Regularization (Weight Decay)
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,eta_min=0.00001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model_filename = f"MEL+STT+ATT_ohneschedluer_Train+val{timestamp}.pth"
    
    train_model(model, train_loader, test_loader, criterion, optimizer,None, num_epochs=num_epochs,best_model_filename=best_model_filename)
    model.load_state_dict(torch.load(os.path.join('models', best_model_filename)))
    #path = "models\MEL+ATT_ohneschedluer_Train+val20250210-190753.pth"
    #model = initialize_mobilenetV3(num_classes=2, dropout = 0.3, input_channels=2)
    #model.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device('cpu')))
    #model.to(device) 
    test_loss, test_metrics = evaluate_model_metrics(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")