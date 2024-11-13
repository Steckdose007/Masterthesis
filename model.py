# model.py
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN1D, self).__init__()
        
        # Convolutional Layers with Batch Normalization to counter
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # Batch Norm for conv1
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)  # Batch Norm for conv2
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)  # Batch Norm for conv3
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
        
        #1D CNN layers with ReLU activation and MaxPooling
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 16, length//2]
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 32, length//4]
        x = self.pool(self.relu(self.conv3(x)))  # [batch_size, 64, length//8]
        #1D CNN layers with Batch Normalization, ReLU activation, and MaxPooling
        # x = self.pool(self.relu(self.bn1(self.conv1(x))))  # [batch_size, 32, length//2]
        # x = self.pool(self.relu(self.bn2(self.conv2(x))))  # [batch_size, 64, length//4]
        # x = self.pool(self.relu(self.bn3(self.conv3(x))))  # [batch_size, 128, length//8]
        
        # Dropout after the last convolutional layer
        x = self.dropout(x)
        # Flatten the output from the CNN for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64 * (length//8)]
        
        # Fully connected layers with ReLU and Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer
        # Apply softmax to get probabilities
        #x = torch.softmax(x, dim=1)
        return x
