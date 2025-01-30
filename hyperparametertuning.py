import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import CNNMFCC  , initialize_mobilenet,initialize_mobilenetV3
from audiodataloader import AudioDataLoader, AudioSegment,split_list_after_speaker
from Dataloader_pytorch import AudioSegmentDataset 
import optuna
from optuna.visualization import plot_param_importances,plot_parallel_coordinate
from tqdm import tqdm
from  Dataloader_fixedlist import FixedListDataset
import numpy as np
from optuna.pruners import MedianPruner, PatientPruner
import pickle
import pandas as pd
from create_fixed_list import TrainSegment

# Global dataset variable

def prepare_dataset():
    with open("data_lists/mother_list.pkl", "rb") as f:
        data = pickle.load(f)

    segments_train, segments_val, segments_test= split_list_after_speaker(data)
    
    return segments_train,segments_val

# Define objective function for Optuna
def objective(trial):
    # ===== Hyperparameters to tune =====
    gamma = trial.suggest_float("gamma", 0.5, 0.99)
    step_size = trial.suggest_int("step_size", 5, 50)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    dropout_rate = trial.suggest_float("dropout",0.1,0.6)
    
    # (Optional) If using SGD, tune momentum as well
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.95)
    else:
        momentum = 0.0  # Not used, but defined for clarity
        weight_decay = trial.suggest_float("weight_decay",1e-5, 1e-3, log=True)


    # ===== Device configuration =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== Data Loaders (example) =====
    # Assuming you have already defined `segments_train`, `segments_val`
    # which are instances of some Dataset. 
    segments_train, segments_val = prepare_dataset()
    segments_train = FixedListDataset(segments_train)
    segments_val = FixedListDataset(segments_val)
    train_loader = DataLoader(segments_train, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True,prefetch_factor=2)  # Fetches 2x the batch size in advance)
    val_loader = DataLoader(segments_val, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True,prefetch_factor=2) 

    # ===== Initialize model =====
    # Example: a simple MobileNet or any other model
    num_classes = 2
    input_channels = 1
    model = initialize_mobilenetV3(num_classes,dropout_rate, input_channels)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)  # Move model to GPU(s)
    # ===== Define loss and optimizer =====
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), 
                              lr=learning_rate, 
                              momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ===== Training loop =====
    num_epochs = 40  
    for epoch in tqdm(range(num_epochs), desc="Processing words"):
        # --- Train ---
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Step the scheduler at the end of each epoch
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == val_labels).sum().item()

        # Compute average validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # Report validation loss to Optuna
        trial.report(val_loss, step=epoch)

        # Optional: If you want to prune based on validation loss
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return final validation loss (or negative accuracy if you want to maximize accuracy)
    return val_loss

def optimize_with_progress(study, objective, n_trials):
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])


if __name__ == "__main__":
    # Create Optuna study
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),direction='minimize', pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=8))
    # Optimize 
    optimize_with_progress(study, objective, n_trials=50)    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters
    with open("best_hyperparameters_stt.txt", "w") as f:
        f.write(f"Best trial accuracy: {trial.value}\n")
        f.write("Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")

    fig = plot_param_importances(study)
    fig.write_image("param_importances_stt.png")
    fig1 = plot_parallel_coordinate(study,target_name="validation loss", include_pruned=True)
    fig1.write_image("plot_parallel_coordinate_stt.png")

    trials_data = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state.name,  # e.g., "COMPLETE", "PRUNED", etc.
        }
        # Add each hyperparameter in t.params
        row.update(t.params)
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    
    # Save to CSV
    df.to_csv("all_trials_results_stt.csv", index=False)