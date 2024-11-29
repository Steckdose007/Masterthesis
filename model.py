# model.py
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN1D, self).__init__()
        kernel_size=5
        padding = 2
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * (input_size // 8), 128)  # Adjust input_size//8 based on pooling
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x should have shape [batch_size, 1, input_length]
        
        # 1D CNN layers with ReLU activation and MaxPooling
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 16, length//2]
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 32, length//4]
        x = self.pool(self.relu(self.conv3(x)))  # [batch_size, 64, length//8]
        
        # Flatten the output from the CNN for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64 * (length//8)]
        
        # Fully connected layers with ReLU and Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer
        # Apply softmax to get probabilities
        #x = torch.softmax(x, dim=1)
        return x
    

    
class CNNMFCC(nn.Module):
    def __init__(self, num_classes, n_mfcc, target_frames, kernel_size=(5, 5), dropout_rate=0.3):
        super(CNNMFCC, self).__init__()
        if(kernel_size==(3, 3)):
            padding=(1,1)
        elif(kernel_size==(5, 5)):
            padding=(2,2)
        elif(kernel_size==(11, 11)):
            padding=(5,5)
        elif(kernel_size==(15, 15)):
            padding=(7,7)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after first conv
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after second conv
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization after third conv

        # Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully Connected Layers
        flattened_size = (n_mfcc // 8) * (target_frames // 8) * 128  # Adjust based on pooling
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Activation
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Shape: (batch, 32, n_mfcc//2, time_frames//2)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Shape: (batch, 64, n_mfcc//4, time_frames//4)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # Shape: (batch, 128, n_mfcc//8, time_frames//8)
        x = self.dropout(x)
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
