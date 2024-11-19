import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D 
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset  
from sklearn.model_selection import train_test_split
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10,best_model_filename = None):
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

if __name__ == "__main__":
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)

    # Load preprocessed audio segments from a pickle file
    words_segments = loader.load_segments_from_pickle("all_words_downsampled_to_24kHz.pkl")
    segments_train, segments_test = train_test_split(words_segments, random_state=42, test_size=0.20)

    # Set target length for padding/truncation
    # maximum word lenght is 65108 and because a strechtching of up to 120% can appear the buffer hast to be that big.
    target_length_8kHz = int(1.2*11811) 
    target_length_16kHz = int(1.2*23622)  
    target_length_24kHz = int(1.2*35433)  
    target_length_32kHz = int(1.2*47244)  
    target_length_44kHz = int(1.2*65108) 
    target_length_24kHz_MFCC = int(35433)#data augmentstion already done
    target_length = target_length_24kHz


    # Hyperparameters
    num_classes = 2  # Adjust based on your classification task (e.g., binary classification for sigmatism)
    learning_rate = 0.001
    num_epochs = 15
    batch_size = 32
    # Create dataset 
    segments_test = AudioSegmentDataset(segments_test, target_length, augment= False)
    segments_train = AudioSegmentDataset(segments_train, target_length, augment = True)

    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(segments_test, batch_size=batch_size, shuffle=False)


    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)
    model = CNN1D(num_classes,target_length).to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_model_filename = f"best_cnn1d_model_{timestamp}.pth"
    
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs,best_model_filename=best_model_filename)
