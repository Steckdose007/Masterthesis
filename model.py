# model.py
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        
        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        
        # Pooling Layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 36499, 128)  # Adjust the dimension based on input length
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
