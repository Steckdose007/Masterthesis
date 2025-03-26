import pickle
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve , auc
import sklearn.metrics
import seaborn as sns
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audiodataloader import split_list_after_speaker

def load_per_word_auc(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def pairwise_comparison_aggregate(logits, target_vocab_index):
    """
    Compute an aggregate score for the target class by comparing its logit
    with the logits of all other classes in each time step.
    """
    logits = np.array(logits)  # Ensure numpy array
    target_logits = logits[:, target_vocab_index]  # Extract logits for the target class (e.g., 's')
    
    # Compute pairwise differences for all other classes
    differences = target_logits[:, np.newaxis] - logits  # Broadcast target class over all classes
    
    # Aggregate differences by summing or averaging across all other classes
    aggregate_score = np.sum(differences, axis=1)  # Sum differences for each time step
    
    return aggregate_score

def linear_normalization_with_sum_to_one(logits):
    logits = np.array(logits)
    min_vals = np.min(logits, axis=1, keepdims=True)
    max_vals = np.max(logits, axis=1, keepdims=True)
    
    # Perform Min-Max scaling
    scaled_logits = (logits - min_vals) / (max_vals - min_vals + 1e-9)  # Avoid division by zero
    
    # # Normalize so each row sums to 1
    # row_sums = np.sum(scaled_logits, axis=1, keepdims=True)
    # normalized_logits = scaled_logits / (row_sums + 1e-9)  # Avoid division by zero
    
    return scaled_logits

def print_outputs(output,label_word,label_path):
    
    logits = output # shape: [time_steps, vocab_size]
    """plot heatmap for all character"""
    # 1. Convert to numpy array for plotting
    logits_np = logits  # shape: (time_steps, vocab_size)
    # 2. Plot as a heatmap
    plt.figure(figsize=(12, 6))
    # We transpose so:
    #  - x-axis = time steps
    #  - y-axis = vocab indices
    # shape becomes (vocab_size, time_steps)
    plt.imshow(logits_np.T, aspect='auto', cmap='plasma', origin='lower')
    # Get vocabulary tokens
    vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))

    # Number of ticks (this will match vocab_size)
    num_vocab = len(vocab_tokens)

    # Set up the ticks on Y-axis at intervals (be careful with large vocab)
    plt.yticks(
        ticks=np.arange(num_vocab),
        labels=vocab_tokens,
        fontsize=6  # might need to reduce font size if it's a large vocab
    )
    plt.title(f"{label_word} Logits Heatmap with {label_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("Vocab Index")
    plt.colorbar(label="Logit Value")
    plt.tight_layout()
    
def modified_softmax_2d(logits, transformation='log'):
    logits = np.array(logits)
    if transformation == 'log':
        logits = np.log(np.maximum(logits, 0) + 1)  # Log transform
    elif transformation == 'sqrt':
        logits = np.sqrt(np.maximum(logits, 0))  # Square-root transform
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def temperature_scaled_softmax_2d(logits, temperature=4.0):
    """
    Apply temperature-scaled softmax to a 2D array of logits.
    
    Args:
        logits (np.ndarray): 2D array with shape [time_steps, vocab_size].
        temperature (float): Temperature scaling factor (T > 0). Higher T smooths probabilities.
    
    Returns:
        np.ndarray: 2D array of softmax probabilities with the same shape as logits.
    """
    logits = np.array(logits)
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Subtract max for numerical stability (row-wise)
    scaled_logits = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    
    # Compute softmax
    exp_logits = np.exp(scaled_logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return probabilities

def plot_bimodal_dist(data):
    dict = {
        "s" : 25,
        "z" : 32,
        "x" : 30,
        "n": 20
    }
    print("Start")
    normal = []
    sigmatism = []
    for entry in data:
        # We only use 'auc' as a single numeric feature here
        """Uswe the raw logits"""
        for char in ["x","s","z"]:
            if char in entry["label"]:
                num_time = np.shape(entry["heatmap"])[0]
                num_s = entry["label"].count('s')
                if(num_s == 0):
                    num_s = 1
                #print(entry["heatmap"].shape)
                logits = entry["heatmap"].cpu().numpy()
                #logits += 25
                #print_outputs(logits,entry["label"],entry["label_path"])
                logits_norm = temperature_scaled_softmax_2d(logits,dict[char])
                #print_outputs(logits_norm,entry["label"],entry["label_path"])
                #print(np.shape(logits_norm))
                #print(np.sum(logits_norm[0,:]))
                #plt.show()
                #print(np.shape(entry["heatmap"]),np.shape(logits))
                auc = (np.sum(logits_norm[:, dict[char]]))/num_s/num_time
                #auc = np.sum(logits_norm)#/num_s/num_time
                label_str = entry["label_path"]
                if label_str == "normal":
                    normal.append(auc)
                elif label_str == "sigmatism":
                    sigmatism.append(auc)
                else:
                    # If there's a different label, you can skip or handle differently
                    continue
        
        """Old approach """
        # num_s = entry["label"].count('s')
        # num_time = np.shape(entry["heatmap"])[0]
        # if(num_s == 0):
        #     num_s = 1
        # auc = (entry["auc"]/num_s)/num_time
        
        # Convert label to 0 or 1
        

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

    plt.title('pairwise_comparison_aggregate AUC Distributions: Normal vs. Sigmatism (KDE) with Means')
    plt.xlabel('AUC')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_logistic_regression(data):
    """
    Given the list of dicts (each with "auc" and "label_path"),
    trains a logistic regression classifier to predict sigmatism vs normal.
    """

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

        auc = entry["auc"]/num_time/num_s
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

        auc = entry["auc"]/num_time/num_s
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
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    # Compute Sensitivity (Recall for positive class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Compute Specificity (Recall for negative class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Sensitivity (Recall for sigmatism): {sensitivity:.3f}")
    print(f"Specificity (Recall for normal): {specificity:.3f}")


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

def naive_threshold(data):

    segments_train, segments_val, segments_test= split_list_after_speaker(data)
    
    # Prepare feature matrix X and label vector y
    X_train = []
    y_train = []
    normal = []
    sigmatism = []
    for entry in segments_train:
        # We only use 'auc' as a single numeric feature here
        num_s = entry["label"].count('s')
        if(num_s == 0):
            num_s = 1

        num_time = np.shape(entry["heatmap"])[0]
        logits = entry["heatmap"].cpu().numpy()
        logits_norm = pairwise_comparison_aggregate(logits,25)
        auc = np.sum(logits_norm)/num_s/num_time
        X_train.append(auc)
        
        # Convert label to 0 or 1
        label_str = entry["label_path"]
        if label_str == "normal":
            normal.append(auc)
            y_train.append(0)
        elif label_str == "sigmatism":
            sigmatism.append(auc)
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

        logits = entry["heatmap"].cpu().numpy()
        logits_norm = pairwise_comparison_aggregate(logits,25)
        auc = np.sum(logits_norm)/num_s/num_time
        X_val.append(auc)
        
        # Convert label to 0 or 1
        label_str = entry["label_path"]
        if label_str == "normal":
            normal.append(auc)
            y_val.append(0)
        elif label_str == "sigmatism":
            sigmatism.append(auc)
            y_val.append(1)
        else:
            # If there's a different label, you can skip or handle differently
            continue
    """Plot the distribution"""
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


    print(np.shape(X_train))
    # Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, 1)  # shape (n_samples, 1)
    y_train = np.array(y_train)
    X_val = np.array(X_val).reshape(-1, 1)  # shape (n_samples, 1)
    y_val = np.array(y_val)
    print(np.shape(X_train))
    #hyperparameter_tuning_logreg(X_train, X_val, y_train, y_val)

    # Train logistic regression model
    #For not normalized:
    clf = LogisticRegression(solver="liblinear",C=0.001, penalty="l2")
    #For double normalized:
    #clf = LogisticRegression(solver="liblinear",C=0.1, penalty="l2")

    clf.fit(X_train, y_train)
    # Evaluate on the test set
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    # Predict probabilities
    y_val_scores = clf.predict_proba(X_val)[:, 1]


    # Compute ROC curve and AUC for validation data
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_scores)
    roc_auc_val = sklearn.metrics.auc(fpr_val, tpr_val)

    # Compute Sensitivity (Recall for positive class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Compute Specificity (Recall for negative class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Sensitivity (Recall for sigmatism): {sensitivity:.3f}")
    print(f"Specificity (Recall for normal): {specificity:.3f}")

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_val, tpr_val, label=f"Validation ROC curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random guess (AUC = 0.50)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def hits_above(data):
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")  
    # -2.215189873417721
    # Inter_dist 1.9240623244177293
    a = np.arange(0, 1,100)
    thresholds = []
    for threshold in a:
        normal = []
        sigmatism = []
        for entry in data:
            # Only process entries that have 's' in their label (word)
            if "s" in entry["label"].lower():  
                # Get logits (shape [time_steps, vocab_size])
                # If it's on GPU, you may want to do: logits = entry["heatmap"].cpu()
                logits = entry["heatmap"]  
                
                # # Convert to probabilities
                probs = F.softmax(logits, dim=-1)  # shape: [time_steps, vocab_size]
                
                # Extract the probabilities for 's' at each time step
                s_probs = probs[:, s_token_id]  # shape: [time_steps]
                """Hits counted above threshold"""
                hits_count=0
                # Count how many frames exceed the threshold
                hits_count = (s_probs >= threshold).sum().item()
                """P_max"""
                #p_max = s_probs.max().item()
                label_str = entry["label_path"]
                if label_str == "normal":
                    normal.append(hits_count)
                elif label_str == "sigmatism":
                    sigmatism.append(hits_count)


        mean_normal = np.mean(normal)
        mean_sigmatism = np.mean(sigmatism)
        thresholds.append(mean_normal-mean_sigmatism)
        print("Inter class mean dist: ",mean_normal-mean_sigmatism, threshold)
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
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    per_word_auc_data = load_per_word_auc("data_lists\STT_heatmap_list.pkl")
    # leng =0
    # for entry in per_word_auc_data:
    #     time = np.shape(entry["heatmap"])[0]
    #     if time > leng:
    #         leng = time
    #         print(leng)
    plot_bimodal_dist(per_word_auc_data)
    # print(leng)
    # 2) Train logistic regression
    #hits_above(per_word_auc_data)
    #naive_threshold(per_word_auc_data)
    #model = train_logistic_regression(per_word_auc_data)

  