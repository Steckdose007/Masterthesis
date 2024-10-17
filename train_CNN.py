import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN1D  # Ensure this points to your CNN model definition
from audiodataloader import AudioDataLoader, AudioSegmentDataset  # Adjust based on actual file path


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


if __name__ == "__main__":
    # Load your dataset
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=True)

    # Load preprocessed audio segments from a pickle file
    words_segments = loader.load_segments_from_pickle("words_segments.pkl")

    # Set target length for padding/truncation
    target_length = 291994  # Adjust this based on your data length

    # Create dataset and DataLoader
    audio_dataset = AudioSegmentDataset(words_segments, target_length)
    train_loader = DataLoader(audio_dataset, batch_size=16, shuffle=True)

    # Hyperparameters
    num_classes = 2  
    learning_rate = 0.001
    num_epochs = 10

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D(num_classes).to(device)  # Ensure num_classes matches your problem
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), 'cnn1d_model.pth')
    print("Model saved successfully.")
