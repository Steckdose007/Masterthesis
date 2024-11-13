import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import CNN1D  # Ensure this points to your CNN model definition
from audiodataloader import AudioDataLoader, AudioSegment
from Dataloader_pytorch import AudioSegmentDataset  # Adjust based on actual file path
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_param_importances,plot_param_importances
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Global dataset variable
global segments_train, segments_test
# Load the dataset once
def prepare_dataset():
    global segments_train, segments_test
    loader = AudioDataLoader(config_file='config.json', word_data=False, phone_data=False, sentence_data=False, get_buffer=False)
    # Load preprocessed audio segments from a pickle file
    words_segments = loader.load_segments_from_pickle("all_words_segments.pkl")
    # Set target length for padding/truncation
    target_length = int(1.2*65108) # maximum word lenght is 65108 and because a strechtching of up to 129% can appear the buffer hast to be that big. 
    
    # Create dataset  
    segments_train, segments_test = train_test_split(words_segments, random_state=42,test_size=0.20)
    segments_train, segments_test = train_test_split(segments_test, random_state=42,test_size=0.20)#small subset
    segments_test = AudioSegmentDataset(segments_test, target_length, augment= False)
    segments_train = AudioSegmentDataset(segments_train, target_length,augment = True)
    return segments_train,segments_test

   

    return train_loader, test_loader
# Define objective function for Optuna
def objective(trial):
    global segments_train, segments_test

    # Define search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Use log=True for logarithmic scale
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(segments_test, batch_size=batch_size, shuffle=False)
   
    target_length = int(1.2*65108) # maximum word lenght is 65108 and because a strechtching of up to 129% can appear the buffer hast to be that big. 
    
    # Define model
    model = CNN1D(num_classes=2, input_size=target_length)
    model.dropout = nn.Dropout(dropout_rate)  # Update dropout rate
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
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
    with open("best_hyperparameters.txt", "w") as f:
        f.write(f"Best trial accuracy: {trial.value}\n")
        f.write("Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")

    fig = plot_param_importances(study)
    fig.show()
    param_importance_plot = plot_param_importances(study)
    param_importance_plot.show()