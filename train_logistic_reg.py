import pickle
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def split_list_after_speaker(words_segments):
    """
    Groups words to their corresponding speakers and creates train test val split
    Returns:
    Train test val split with speakers
    """
    # Group word segments by speaker
    speaker_to_segments = defaultdict(list)
    for segment in words_segments:
        normalized_path = segment["path"].replace("\\", "/")
        #print(normalized_path)
        _, filename = os.path.split(normalized_path)
        #print(filename)
        speaker = filename.replace('_sig', '')
        #print(speaker)
        speaker_to_segments[speaker].append(segment)
    # Get a list of unique speakers
    speakers = list(speaker_to_segments.keys())
    print("number speakers: ",np.shape(speakers))
    # Split speakers into training and testing sets
    speakers_train, speakers_test = train_test_split(speakers, random_state=42, test_size=0.13)
    speakers_train, speakers_val = train_test_split(speakers_train, random_state=42, test_size=0.07)

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

def load_per_word_auc(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def train_logistic_regression(data):
    """
    Given the list of dicts (each with "auc" and "label_path"),
    trains a logistic regression classifier to predict sigmatism vs normal.
    """
    dict = {
        "s" : 25,
        "z" : 32,
        "x" : 30,
        "n": 20
    }
    
    normal = []
    sigmatism = []
    for entry in data:
        # We only use 'auc' as a single numeric feature here
        # for char in ["x","s","z"]:
        #     if char in entry["label"]:
        #         num_time = np.shape(entry["heatmap"])[0]
        #         logits = entry["heatmap"].cpu().numpy()[:, dict[char]]
        #         #logits += 25
        #         #print(np.shape(entry["heatmap"]),np.shape(logits))
        #         auc = np.sum(logits)/num_time

        num_s = entry["label"].count('s')
        num_time = np.shape(entry["heatmap"])[0]
        if(num_s == 0):
            num_s = 1
        auc = (entry["auc"]/num_s)/num_time
        
        # Convert label to 0 or 1
        label_str = entry["label_path"]
        if label_str == "normal":
            normal.append(auc)
        elif label_str == "sigmatism":
            sigmatism.append(auc)
        else:
            # If there's a different label, you can skip or handle differently
            continue

    mean_normal = np.mean(normal)
    mean_sigmatism = np.mean(sigmatism)

    plt.figure(figsize=(8, 6))

    # You can pick your own colors or let seaborn choose:
    normal_color = 'blue'
    sigmatism_color = 'orange'

    # Plot KDE for Normal
    sns.kdeplot(normal, shade=True, color=normal_color, label='Normal')
    # Plot KDE for Sigmatism
    sns.kdeplot(sigmatism, shade=True, color=sigmatism_color, label='Sigmatism')

    # Draw vertical lines for the means
    plt.axvline(mean_normal, color=normal_color, linestyle='--',
                label=f'Mean Normal = {mean_normal:.2f}')
    plt.axvline(mean_sigmatism, color=sigmatism_color, linestyle='--',
                label=f'Mean Sigmatism = {mean_sigmatism:.2f}')

    plt.title('AUC Distributions: Normal vs. Sigmatism (KDE) with Means')
    plt.xlabel('AUC')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()




    segments_train, segments_val, segments_test= split_list_after_speaker(data)
    # Prepare feature matrix X and label vector y
    X_train = []
    y_train = []
    for entry in segments_train:
        # We only use 'auc' as a single numeric feature here
        num_s = entry["label"].count('s')
        if(num_s == 0):
            num_s = 1

        num_time = np.shape(entry["heatmap"])[0]

        auc = entry["auc"]/num_time
        X_train.append(auc)
        
        # Convert label to 0 or 1
        label_str = entry["label_path"]
        if label_str == "normal":
            y_train.append(0)
        elif label_str == "sigmatism":
            y_train.append(1)
        else:
            # If there's a different label, you can skip or handle differently
            continue
    
    X_val = []
    y_val = []    
    for entry in segments_val:
        # We only use 'auc' as a single numeric feature here
        num_s = entry["label"].count('s')
        if(num_s == 0):
            num_s = 1
        num_time = np.shape(entry["heatmap"])[0]

        auc = entry["auc"]/num_time
        X_val.append(auc)
        
        # Convert label to 0 or 1
        label_str = entry["label_path"]
        if label_str == "normal":
            y_val.append(0)
        elif label_str == "sigmatism":
            y_val.append(1)
        else:
            # If there's a different label, you can skip or handle differently
            continue

 

    print(np.shape(X_train))
    # Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, 1)  # shape (n_samples, 1)
    y_train = np.array(y_train)
    X_val = np.array(X_val).reshape(-1, 1)  # shape (n_samples, 1)
    y_val = np.array(y_val)
    print(np.shape(X_train))

    hyperparameter_tuning_logreg(X_train, X_val, y_train, y_val)

    # Create and train the logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    return clf

def hyperparameter_tuning_logreg(X_train, X_val, y_train, y_val):
    """
    Performs hyperparameter tuning for logistic regression using GridSearchCV.
    Returns the best estimator and prints performance metrics.
    """
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    # Define parameter grid
    param_grid = {
        'solver': ['liblinear', 'saga'],  # solvers that allow for L1 or L2
        'penalty': ['l1', 'l2'],          # which penalty to use
        'C': [0.001, 0.01, 0.1, 1, 10, 100]  # regularization strength
    }

    # Create logistic regression model
    lr = LogisticRegression()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        lr, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=10,             # 5-fold cross-validation
        verbose=1         # just to see some progress logs
    )

    # Fit GridSearchCV
    grid_search.fit(X_trainval, y_trainval)
    print("\nBest Params:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)


def hits_above(data):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")  
    # -2.215189873417721
    # Inter_dist 1.9240623244177293
    threshold=5
    normal = []
    sigmatism = []
    for entry in data:
        # Only process entries that have 's' in their label (word)
        if "s" in entry["label"].lower():  
            # Get logits (shape [time_steps, vocab_size])
            # If it's on GPU, you may want to do: logits = entry["heatmap"].cpu()
            logits = entry["heatmap"]  
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: [time_steps, vocab_size]
            
            # Extract the probabilities for 's' at each time step
            s_probs = probs[:, s_token_id]  # shape: [time_steps]
            #hits_count=0
            # Count how many frames exceed the threshold
            #hits_count = (s_probs >= threshold).sum().item()
            p_max = s_probs.max().item()

            label_str = entry["label_path"]
            if label_str == "normal":
                normal.append(p_max)
            elif label_str == "sigmatism":
                sigmatism.append(p_max)


    mean_normal = np.mean(normal)
    mean_sigmatism = np.mean(sigmatism)
    # You can pick your own colors or let seaborn choose:
    normal_color = 'blue'
    sigmatism_color = 'orange'

    # Plot KDE for Normal
    sns.kdeplot(normal, shade=True, color=normal_color, label='Normal')
    # Plot KDE for Sigmatism
    sns.kdeplot(sigmatism, shade=True, color=sigmatism_color, label='Sigmatism')

    # Draw vertical lines for the means
    plt.axvline(mean_normal, color=normal_color, linestyle='--',
                label=f'Mean Normal = {mean_normal:.2f}')
    plt.axvline(mean_sigmatism, color=sigmatism_color, linestyle='--',
                label=f'Mean Sigmatism = {mean_sigmatism:.2f}')

    plt.title(f'Max P for s: Normal vs. Sigmatism (KDE) with Means')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1) Load data from pickle
    per_word_auc_data = load_per_word_auc("STT_csv\per_word_auc_values.pkl")
    # leng =0
    # for entry in per_word_auc_data:
    #     time = np.shape(entry["heatmap"])[0]
    #     if time > leng:
    #         leng = time
    #         print(leng)

    # print(leng)
    # 2) Train logistic regression
    hits_above(per_word_auc_data)
    #model = train_logistic_regression(per_word_auc_data)

  