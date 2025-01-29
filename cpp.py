import numpy as np
import scipy.signal 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from acoustics.cepstrum import complex_cepstrum
from acoustics.cepstrum import real_cepstrum
from acoustics.cepstrum import inverse_complex_cepstrum
from acoustics.cepstrum import minimum_phase
from audiodataloader import AudioDataLoader, AudioSegment
import os
import pickle
import random
from Dataloader_pytorch import AudioSegmentDataset ,process_and_save_dataset
import pandas as pd
from scipy import linalg
import scipy.stats as stats
import warnings
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from frechet_audio_distance  import FrechetAudioDistance  

def compute_mean_and_cov(features: np.ndarray):
    """
    Compute the mean vector and covariance matrix of the given set of feature vectors.
    features shape: (n_samples, embed_dim)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)  # rowvar=False => each row is an observation
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Fréchet Distance (FID metric):
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))

    Parameters
    ----------
    mu1, sigma1 : Mean vector and covariance of distribution 1
    mu2, sigma2 : Mean vector and covariance of distribution 2
    eps         : Small constant added to the diagonal if covariance is nearly singular

    Returns
    -------
    fid_value   : Floating-point scalar (lower is more similar)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    #print("Convmean sqrtm")
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(
            "FID calculation produced singular product; adding epsilon to the diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    # Numerical error might cause slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real


    fid_value = (
        diff.dot(diff)# = ||mu1 - mu2||^2
        + np.trace(sigma1 + sigma2 - 2 * covmean)
    )
    return fid_value

def extract_phone_segments(segment):
    """Approach 2
    Returns:
        extracted and conncatenated frames of phones of intrest from a segment.
    """
    audio_signal = segment.audio_data
    segment_label = segment.label
    #print(segment_label)
    phone_chars=['s','S','Z', 'z','X', 'x','Ÿ']#,'Ã'
    # Get the total duration of the audio signal in seconds
    sr=24000
    total_duration = len(audio_signal) / sr

    # Normalize word length into equal time parts
    num_chars = len(segment_label)
    time_per_char = total_duration / num_chars

    # Find positions of the phone characters in the word
    phone_positions = [i for i, char in enumerate(segment_label) if char in phone_chars]
    #print(phone_positions)
    # Map phone positions to time intervals
    audio_segments = []
    for position in phone_positions:
        char_start_time = position * time_per_char
        char_end_time = (position + 1) * time_per_char
        if char_start_time < 0 or char_end_time > len(audio_signal) or char_start_time == char_end_time:
            continue
        char_start_time = int(char_start_time * sr)
        char_end_time = int(char_end_time * sr)
        audio_segments.append(audio_signal[char_start_time:char_end_time])
    combined_audio = np.concatenate(audio_segments, axis=0)
    
    return combined_audio

def cpp_calc_and_plot(x, fs, pitch_range, trendline_quefrency_range, cepstrum,plotting = False):
    
    # Cepstrum
    if cepstrum == 'complex_cepstrum':
        ceps, _ = complex_cepstrum(x)
    elif cepstrum == 'real_cepstrum':
        ceps = real_cepstrum(x)
    elif cepstrum == 'vfp':
        x = np.hamming(len(x))*x
        spectrum = np.fft.rfft(x)
        spectrum = 20*np.log10(np.abs(spectrum))
        ceps = np.fft.rfft(spectrum) 
    elif cepstrum == 'cepstrum':
        spectrum = np.fft.fft(x)
        spectrum = 20*np.log10(np.abs(spectrum))
        ceps = np.fft.fft(spectrum) 
    ceps = 20*np.log10(np.abs(ceps))

    # Quefrency
    dt = 1/fs
    freq_vector = np.fft.rfftfreq(len(x), d=dt)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(2*ceps.size-2, df)
    
    # Selecting part of cepstrum
    quefrency_range = [1/pitch_range[1], 1/pitch_range[0]]
    index_range = np.where((quefrency_vector >= quefrency_range[0]) & (quefrency_vector <=quefrency_range[1]))

    # For trend line
    index_range_tl = np.where((quefrency_vector >= trendline_quefrency_range[0]) & (quefrency_vector <=trendline_quefrency_range[1]))
    plot_range = np.where((quefrency_vector >= 0) & (quefrency_vector <= 0.0512))
    
    # Linear regression
    linear_regressor = LinearRegression()  
    linear_regressor.fit(quefrency_vector[index_range_tl].reshape(-1, 1), ceps[index_range_tl].reshape(-1, 1))  
    Y_pred = linear_regressor.predict(quefrency_vector.reshape(-1, 1))  
    
    peak_value = np.max(ceps[index_range])
    peak_index = np.argmax(ceps[index_range])
    if plotting:
        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.plot(quefrency_vector[plot_range], ceps[plot_range])
        ax.plot(quefrency_vector[plot_range], Y_pred[plot_range])
        
        ax.plot(quefrency_vector[index_range][peak_index], peak_value, marker="o",markeredgecolor="red", markerfacecolor="red")
        ax.plot(quefrency_vector[index_range][peak_index], Y_pred[index_range][peak_index], marker="o", markeredgecolor="red", markerfacecolor="red")
        ax.set_xlabel('quefrency[s]')
        ax.set_ylabel('log magnitude(dB)')
        ax.set_title('Cepstrum')
    
        print('The peak is found at quefrency {}s and its value is {}'.format(np.round(quefrency_vector[index_range][peak_index], 5), np.round(peak_value, 5)))
        print('The trendline value at this quefrency is {}'.format(np.round(Y_pred[index_range][peak_index][0], 5)))
        print('The CPP is {} dB'.format(np.round(peak_value - Y_pred[index_range][peak_index][0], 5)))
        plt.show()
    return np.round(peak_value - Y_pred[index_range][peak_index][0], 5)

def get_cppplots_per_speaker_and_disorder(words_segments,phones = None):
    """CPP for normal and sigmatism 
    plots for all words
    plots devided per speaker
    """
    sigmatism = []
    normal = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        #print(filename1)
        extracted = extract_phone_segments(word)
        cpp_peak = cpp_calc_and_plot(extracted,word.sample_rate,pitch_range=[60, 8000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
        if word.label_path == "sigmatism":
            sigmatism.append((20 * np.log10(cpp_peak)))
        else:
            normal.append((20 * np.log10(cpp_peak)))

    data = [sigmatism, normal]
    labels = ['Sigmatism', 'Normal']

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, notch=True, showmeans=True)

    # Customize the appearance
    plt.title('CPP Distribution for Sigmatism and Normal Words', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('CPP (dB)', fontsize=12)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

    """CPP per word per speaker"""
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        if(word.label_path == "sigmatism"):
            filename1 = filename1.replace("_sig", "")
        extracted = extract_phone_segments(word)
        cpp_peak = cpp_calc_and_plot(extracted,word.sample_rate,pitch_range=[60, 8000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
        data.append({'Speaker': filename1, 'Word': word.label, 'Category': word.label_path, 'CPP': (20 * np.log10(cpp_peak))})

    df = pd.DataFrame(data)
    print(df.head())
    # Group by Speaker
    unique_speakers = df['Speaker'].unique()
    # Group speakers into batches of 20
    batch_size = 10
    batches = [unique_speakers[i:i + batch_size] for i in range(0, len(unique_speakers), batch_size)]

    # Plot each batch
    for i, batch in enumerate(batches):
        plt.figure(figsize=(12, 10))  # Adjust size for horizontal plot
        
        # Filter data for the current batch of speakers
        batch_data = df[df['Speaker'].isin(batch)]
        
        # Horizontal boxplot data
        categories = ['normal', 'sigmatism']
        data = [batch_data[(batch_data['Speaker'] == speaker) & (batch_data['Category'] == cat)]['CPP']
                for speaker in batch for cat in categories]
        
        labels = [f"{speaker} ({cat})" for speaker in batch for cat in categories]
        
        plt.boxplot(data, labels=labels, patch_artist=True, notch=True, showmeans=True, vert=False)
        plt.title(f"CPP Distribution for Speakers Batch {i + 1}")
        plt.xlabel("CPP (dB)")
        plt.ylabel("Speakers (Normal and Sigmatism)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

def paired_t_test(words_segments):
    """CPP per word per speaker all plottet in a plot to see the correlation"""
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        if(word.label_path == "sigmatism"):
            filename1 = filename1.replace("_sig", "")
        #extracted = extract_phone_segments(word)
        cpp_peak = cpp_calc_and_plot(word.audio_data,word.sample_rate,pitch_range=[60, 8000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
        data.append({'Speaker': filename1, 'Category': word.label_path, 'CPP': (20 * np.log10(cpp_peak))})

    df = pd.DataFrame(data)
    print(df.head())
    # Group by Speaker
    unique_speakers = df['Speaker'].unique()
    # 1. Aggregate to get the mean CPP for each speaker in both categories
    df_normal = df[df['Category'] == 'normal'].groupby('Speaker')['CPP'].mean().reset_index()
    df_sigmatism = df[df['Category'] == 'sigmatism'].groupby('Speaker')['CPP'].mean().reset_index()

    # 2. Merge them side by side so we can do a paired comparison
    df_merged = pd.merge(df_normal, df_sigmatism, on='Speaker', suffixes=('_normal','_sigmatism'))

    # The columns in df_merged will be: Speaker, CPP_normal, CPP_sigmatism
    # Compute the difference, if you want to look at it
    df_merged['Difference'] = df_merged['CPP_normal'] - df_merged['CPP_sigmatism']

    # 3. Paired t-test comparing normal means vs sigmatism means for each speaker
    ttest_result = stats.ttest_rel(df_merged['CPP_normal'], df_merged['CPP_sigmatism'])
    print("Paired t-test result:", ttest_result)

    # 4. Visualization: lines from Normal to Sigmatism per speaker
    #    Sort by normal CPP or sigmatism CPP just so they’re in a nice left→right order on the plot
    df_merged_sorted = df_merged.sort_values(by='CPP_normal')

    # Extract the arrays for plotting
    speakers = df_merged_sorted['Speaker']
    normal_means = df_merged_sorted['CPP_normal']
    sigmatism_means = df_merged_sorted['CPP_sigmatism']

    plt.figure(figsize=(8,6))
    for i in range(len(speakers)):
        # Plot a line from (0, normal) to (1, sigmatism)
        plt.plot([0, 1], [normal_means.iloc[i], sigmatism_means.iloc[i]], 
                marker='o', label='_nolegend_')

    # Make the x-axis show "Normal" vs "Sigmatism" instead of 0 vs 1
    plt.xticks([0, 1], ['Normal', 'Sigmatism'])
    plt.title("Paired Plot of Normal vs. Sigmatism Mean CPP per Speaker")
    plt.ylabel("Mean CPP (dB)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(8,6))
    for i in range(len(speakers)):
        # Plot a line from (0, normal) to (1, sigmatism)
        plt.plot([0, 1],
                [normal_means.iloc[i], sigmatism_means.iloc[i]],
                marker='o', color='gray', linewidth=1, alpha=0.7)

    # 5. Plot the overall (mean) slope in a distinct color
    avg_normal = normal_means.mean()
    avg_sigmatism = sigmatism_means.mean()
    plt.plot([0, 1],
            [avg_normal, avg_sigmatism],
            color='red', marker='o', linewidth=2, label='Overall Mean Slope')

    # Make the x-axis show "Normal" vs "Sigmatism" instead of 0 vs 1
    plt.xticks([0, 1], ['Normal', 'Sigmatism'])
    plt.title("Paired Plot of Normal vs. Sigmatism Mean CPP per Speaker")
    plt.ylabel("Mean CPP (dB)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def fix_audio_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure `audio` has exactly `target_length` samples by
    truncating if too long or zero-padding if too short.
    """
    current_length = len(audio)
    
    if current_length > target_length:
        # Truncate / crop
        return audio[:target_length]
    elif current_length < target_length:
        # Zero-pad
        padded_audio = np.zeros(target_length, dtype=audio.dtype)
        padded_audio[:current_length] = audio
        return padded_audio
    else:
        # Already the correct length
        return audio

def fid_for_two_arrays(array1: np.ndarray, array2: np.ndarray, eps=1e-6) -> float:
    """
    Compute the FID for two arrays (assumed to be feature embeddings).
    """
    #print("Compute mean and cov")
    mu1, sigma1 = compute_mean_and_cov(array1)
    #print("Compute mean2 and cov2")

    mu2, sigma2 = compute_mean_and_cov(array2)
    return frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)  
  
def fid_plotting(words_segments):
    """
    Function to calculate the FID between the normal and sigmatism words per speaker 
    plots FID for every word in a boxplot
    plots mean FID per speaker in a boxplot
    """
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        if(word.label_path == "sigmatism"):
            filename1 = filename1.replace("_sig", "")
        audio = fix_audio_length(word.audio_data,int(1.7*16000))
        data.append({'Speaker': filename1, 'Category': word.label_path, 'Audio':audio,'WordLabel': word.label })

    df = pd.DataFrame(data)
    print("DataFrame built. Shape =", df.shape)

    
    # 4) We'll store per-speaker FID results in a list
    results = []
    
    # 2) Group by (Speaker, WordLabel)
    grouped = df.groupby(['Speaker', 'WordLabel'])
    #print(grouped.head())
    results = []
    
    for (speaker, wlabel), group in grouped:
        #print(group.head())
        #print(speaker,wlabel)

        # group should have something like:
        #   Category == normal -> 1 row
        #   Category == sigmatism -> 1 row
        normal_rows = group[group['Category'] == 'normal']
        sigmatism_rows = group[group['Category'] == 'sigmatism']
        
        if len(normal_rows) == 0 or len(sigmatism_rows) == 0:
            # e.g. if we're missing a pair
            fid_val = None
        else:
            # Usually you'd have exactly 1 row in normal_rows and 1 in sigmatism_rows.
            # If there's exactly 1 row each, no need to stack. Just take the single array:
            normal_audio = normal_rows['Audio'].values[0]      # shape (audio_length,)
            sigmatism_audio = sigmatism_rows['Audio'].values[0]
            
            # But FID formula expects distributions, i.e. shape (n_samples, embed_dim).
            # So let's just treat each single audio as a 1-sample distribution:
            normal_array = normal_audio[np.newaxis, :]         # shape (1, audio_length)
            sigmatism_array = sigmatism_audio[np.newaxis, :]
            
            fid_val = fid_for_two_arrays(normal_array, sigmatism_array)

        results.append({
            'Speaker': speaker,
            'WordLabel': wlabel,
            'FID': fid_val
        })
    results_df = pd.DataFrame(results)
    #print("Per-Speaker FID:\n", results_df)
    df_clean = results_df.dropna(subset=['FID'])
    
    plt.figure(figsize=(6, 6))
    plt.boxplot(df_clean['FID'],patch_artist=True, showmeans=True)
    plt.ylabel("FID (normal vs. sigmatism)")
    plt.title("Distribution of FID Scores Across All Speakers")
    plt.show()

    speaker_means_df = df_clean.groupby('Speaker')['FID'].mean().reset_index()
    speaker_means_df.columns = ['Speaker', 'MeanFID']

    # Make a new boxplot
    plt.figure(figsize=(6, 6))
    plt.boxplot(speaker_means_df['MeanFID'],patch_artist=True, showmeans=True)
    plt.ylabel("Mean FID (across words)")
    plt.title("Distribution of Mean FIDs Per Speaker")
    plt.show()

def compare_sonne_pairs(words_segments):
    """
    For each speaker, we find the two 'Sonne' normal samples 
    and compute their FID. Then do the same for the two 'Sonne'
    sigmatism samples.

    Finally, we plot the results in a boxplot with two groups:
     - normal 'Sonne' pair FIDs
     - sigmatism 'Sonne' pair FIDs
    """
    # 1) Build a DataFrame from 'words_segments'
    data = []
    for word in words_segments:
        # Extract a speaker ID from the path
        filename1 = os.path.splitext(os.path.basename(word.path))[0]
        
        # If you have a special naming for sigmatism
        # so that speaker name is inside the filename, you might remove "_sig"
        if word.label_path == "sigmatism":
            filename1 = filename1.replace("_sig", "")
        
        # Pad/truncate audio if needed
        audio_fixed = fix_audio_length(word.audio_data, int(1.7 * 16000))
        
        data.append({
            "Speaker": filename1,
            "Category": word.label_path,  # 'normal' or 'sigmatism'
            "Audio": audio_fixed,
            "WordLabel": word.label       # e.g. 'Sonne'
        })
    
    df = pd.DataFrame(data)
    print(df)
    # 2) Filter only rows where WordLabel == 'Sonne'
    df_sonne = df[df["WordLabel"] == "Sonne"].copy()
    print(df_sonne)
    # 3) Group by (Speaker, Category) so we group the 
    #    two normal 'Sonne' clips, and the two sigmatism 'Sonne' clips
    grouped = df_sonne.groupby(["Speaker", "Category"])
    
    # We'll store the result of comparing the 2 "Sonne" samples.
    results = []
    
    for (speaker, cat), group in grouped:
        # We expect exactly 2 rows in each group if the data is consistent
        if len(group) < 2:
            # If a speaker has fewer than 2 samples for this category, skip
            continue
        
        # Let's just compare the first two rows:
        audio1 = group.iloc[0]["Audio"]
        audio2 = group.iloc[1]["Audio"]
        
        # Turn each audio (length N) into shape (1, N) 
        # so that your fid function sees them as 1-sample distributions
        audio1_2d = audio1[np.newaxis, :]
        audio2_2d = audio2[np.newaxis, :]

        fid_val = fid_for_two_arrays(audio1_2d, audio2_2d)

        results.append({
            "Speaker": speaker,
            "Category": cat,   # 'normal' or 'sigmatism'
            "FID": fid_val
        })
    
    results_df = pd.DataFrame(results)
    print("Comparison among the two 'Sonne' samples per category:\n", results_df)

    # 4) Boxplot: Compare normal vs. sigmatism FID across all speakers
    plt.figure(figsize=(8, 6))
    plt.title("FID of the Two 'Sonne' Samples per Speaker (Normal vs. Sigmatism)")
    
    # We can manually pass [normal_FIDs, sigmatism_FIDs] to plt.boxplot
    normal_fids = results_df[results_df["Category"] == "normal"]["FID"]
    sig_fids     = results_df[results_df["Category"] == "sigmatism"]["FID"]
    
    plt.boxplot([normal_fids, sig_fids], 
                labels=["Normal 'Sonne' Pair", "Sigmatism 'Sonne' Pair"],
                patch_artist=True, showmeans=True)
    
    plt.ylabel("FID")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def fid_plotting_randompairs(words_segments, n_pairs=10):
    """
    1) Build a DataFrame of all 'words_segments'
       with columns: [Speaker, Category, Audio, WordLabel].
    2) Randomly pick 'n_pairs' pairs of rows (distinct).
    3) Fix audio length, shape them to (1, length).
    4) Compute FID for each pair.
    5) Show distribution of FIDs in a boxplot.
    """
    # 1) Build a DataFrame
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]
        audio_fixed = fix_audio_length(word.audio_data, int(1.2 * 24000))
        
        data.append({
            "Speaker": filename1,       # or however you parse the speaker
            "Category": word.label_path, # 'normal' or 'sigmatism'
            "Audio": audio_fixed,
            "WordLabel": word.label     # e.g. 'Sonne' or anything else
        })
    
    df = pd.DataFrame(data)
    print(f"DataFrame built. Shape = {df.shape}")
    
    # 2) Randomly pick n_pairs of distinct row indices
    #    We want pairs of *different* rows, so we can do something like:
    all_indices = list(df.index)
    fid_values = []

    # We'll store info about which rows we compared (optional)
    pairs_info = []

    for _ in range(n_pairs):
        # pick 2 distinct random rows
        pair_indices = random.sample(all_indices, 2)
        
        row1 = df.loc[pair_indices[0]]
        row2 = df.loc[pair_indices[1]]
        
        audio1 = row1["Audio"]
        audio2 = row2["Audio"]
        
        # shape them so fid_for_two_arrays sees them as 1-sample distributions
        audio1_2d = audio1[np.newaxis, :]
        audio2_2d = audio2[np.newaxis, :]
        
        fid_val = fid_for_two_arrays(audio1_2d, audio2_2d)
        fid_values.append(fid_val)
        
        pairs_info.append({
            "Index1": pair_indices[0],
            "Index2": pair_indices[1],
            "Speaker1": row1["Speaker"],
            "Speaker2": row2["Speaker"],
            "Category1": row1["Category"],
            "Category2": row2["Category"],
            "WordLabel1": row1["WordLabel"],
            "WordLabel2": row2["WordLabel"],
            "FID": fid_val
        })

    # Convert pairs info to a DataFrame
    df_pairs = pd.DataFrame(pairs_info)
    print("\nRandom Pairs FID Info:\n", df_pairs)

    # 3) Plot the distribution of FIDs
    plt.figure(figsize=(6,6))
    plt.boxplot(fid_values, patch_artist=True, showmeans=True)
    plt.ylabel("FID (random pairs)")
    plt.title(f"Distribution of FIDs for Random Word Pairs")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def compute_fid_for_heatmap(data, sample_rate=16000):
    """
    Compute FID between normal and sigmatism speech using Wav2Vec2 logits.
    Args:
        words_segments (list): List of word segments with labels 'normal' or 'sigmatism'.
        sample_rate (int): Sample rate of the audio.
    Returns:
        fid: Fréchet distance between normal and sigmatism
    """
    normal_logits = []
    sigmatism_logits = []
    
    for word in tqdm(data, desc="Processing words"):
        label = word["label_path"]
        logits_flat = word["heatmap"].mean(axis=0)  # Shape: (vocab_size,)
        
        # Separate into normal and sigmatism groups
        if label == "normal":
            normal_logits.append(logits_flat)
        elif label == "sigmatism":
            sigmatism_logits.append(logits_flat)

    # Convert to NumPy arrays
    normal_array = np.vstack(normal_logits)
    sigmatism_array = np.vstack(sigmatism_logits)
    print(np.shape(normal_array))
    # Compute mean and covariance for each category
    mu1, sigma1 = compute_mean_and_cov(normal_array)
    mu2, sigma2 = compute_mean_and_cov(sigmatism_array)
    print("mu1 shape:", mu1.shape)
    print("sigma1 shape:", sigma1.shape)
    print("mu2 shape:", mu2.shape)
    print("sigma2 shape:", sigma2.shape)
    # Compute FID
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    print("FID score with heatmap STT:", fid)
    return fid    

def FAD_libary():
    SAMPLE_RATE = 16000  # VGGish and many other models often assume 16kHz
    print("Load model")
    frechet = FrechetAudioDistance(
        ckpt_dir="path/to/vggish",       # Where your VGGish checkpoint resides
        model_name="vggish",             # or "clap", "pann", etc., depending on library support
        sample_rate=SAMPLE_RATE,
        use_pca=False,                   # For VGGish: if True, it applies PCA to embeddings
        use_activation=False,            # For VGGish: if True, it extracts an earlier activation
        verbose=True,
        audio_load_worker=4,             # Number of parallel workers to load audio
        # enable_fusion=False            # Some models allow fusion, e.g., CLAP
    )
    print("Calculate Score")
    # Directories with .wav files
    gen_dir = "Data/sigmatism16kHz"
    ref_dir = "Data/normal16kHz"

    # Ensure they exist and contain wav files
    assert os.path.isdir(ref_dir), f"Directory {ref_dir} not found."
    assert os.path.isdir(gen_dir), f"Directory {gen_dir} not found."
    fad_score = frechet.score(ref_dir, gen_dir)
    print("FAD score with model:", fad_score)

def compute_fid_hidden_features(words_segments, sample_rate=16000, layer=-1):
    """
    Compute FID between normal and sigmatism speech using features from a specific Wav2Vec2 layer.
    Args:
        words_segments (list): List of word segments with labels 'normal' or 'sigmatism'.
        sample_rate (int): Sampling rate of the audio.
        layer (int): Which hidden layer to extract features from (-1 = last feature layer).
    Returns:
        fid: Fréchet distance between normal and sigmatism.
    """
    normal_features = []
    sigmatism_features = []
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    batch_size=64
    # Process in batches
    for i in tqdm(range(0, len(words_segments), batch_size)):
        batch = words_segments[i:i + batch_size]
        batch_features = []
        
        for word in batch:
            audio = word.audio_data
            label = word.label_path  # 'normal' or 'sigmatism'
            
            # Extract padded features
            padded_features = extract_features(audio, sample_rate,processor,model, layer=layer)
            batch_features.append(padded_features.flatten())  # Flatten each feature
            
            # Append to the respective group
            if label == "normal":
                normal_features.append(batch_features[-1])
            elif label == "sigmatism":
                sigmatism_features.append(batch_features[-1])
    
    # Compute mean and covariance for normal and sigmatism
    mu1, sigma1 = compute_mean_and_cov_batch(normal_features)
    mu2, sigma2 = compute_mean_and_cov_batch(sigmatism_features)
    
    # Compute FID
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    print("FID score with hiddenlayer STT batch:", fid)

    return fid

def extract_features(audio: np.ndarray, sample_rate: int,processor, model, layer: int = -1):
    """
    Extract features from a specific layer of the Wav2Vec2 model.
    Args:
        audio (np.ndarray): Input audio signal.
        sample_rate (int): Sampling rate of the audio.
        layer (int): Which layer to extract features from (-1 = last hidden layer).
    Returns:
        features (np.ndarray): Hidden layer features of shape (time_steps, hidden_dim).
    """
    
    # Preprocess audio
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states
    hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer
    features = hidden_states[layer].squeeze(0).numpy()  # Shape: (time_steps, hidden_dim) (35,1024)
    time_steps, hidden_dim = features.shape
    target_time_steps: int = 84
    if time_steps == target_time_steps:
        return features
    elif time_steps > target_time_steps:
        # Truncate to the first `target_time_steps`
        return features[:target_time_steps, :]
    else:
        # Pad with zeros to reach `target_time_steps`
        padding = np.zeros((target_time_steps - time_steps, hidden_dim))
        return np.vstack([features, padding])
    
def compute_mean_and_cov_batch(features_list):
    """
    Compute mean and covariance incrementally from a list of feature arrays.
    Args:
        features_list (list): List of feature arrays for each batch.
    Returns:
        mean (np.ndarray): Mean vector of shape (embed_dim,).
        cov (np.ndarray): Covariance matrix of shape (embed_dim, embed_dim).
    """
    # Initialize variables
    total_samples = 0
    mean_accum = 0
    cov_accum = 0
    
    for features in features_list:
        total_samples += features.shape[0]
        mean_batch = features.mean(axis=0)  # Mean for this batch
        cov_batch = np.cov(features, rowvar=False)  # Covariance for this batch
        
        # Incrementally update the total mean
        mean_accum += mean_batch * features.shape[0]
        
        # Incrementally update the covariance
        cov_accum += cov_batch * features.shape[0]
    
    # Normalize mean and covariance
    mean = mean_accum / total_samples
    cov = cov_accum / total_samples
    
    return mean, cov

def load_per_word_auc(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def split_list_after_speaker(words_segments):
    """
    Groups words to their corresponding speakers and creates train test val split
    Returns:
    Train test val split with speakers
    """
    # Group word segments by speaker
    speaker_to_segments = defaultdict(list)
    for segment in words_segments:
        normalized_path = segment.path.replace("\\", "/")
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
    speakers_train, speakers_test = train_test_split(speakers, random_state=42, test_size=0.05)
    speakers_train, speakers_val = train_test_split(speakers_train, random_state=42, test_size=0.15)

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

if __name__ == "__main__":

    per_word_auc_data = load_per_word_auc("STT_csv\per_word_auc_values.pkl")
    compute_fid_for_heatmap(per_word_auc_data)

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True, downsample=True)
    phones_segments = loader.load_segments_from_pickle("data_lists\phone_normalized_16kHz.pkl")
    words_segments = loader.load_segments_from_pickle("data_lists\words_without_normalization_16kHz.pkl")
    mfcc_dim={
        "n_mfcc":128, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)
    compute_fid_hidden_features(segments_val)
    #segments = AudioSegmentDataset(words_segments,phones_segments, mfcc_dim, augment= False)
    #mu1, sigma1 = compute_mean_and_cov(words_segments[0].audio_data)
    #mu2, sigma2 = compute_mean_and_cov(words_segments[0].audio_data)
    #eps=1e-6
    #fid_value = frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
    #print(fid_value)
    #fid_plotting_randompairs(words_segments,3000)
    #FAD_libary() # use method with model
    #compare_sonne_pairs(words_segments)
    #fid_plotting(words_segments) 
    #paired_t_test(words_segments)
    #word = words_segments[0]
    #cpp_calc_and_plot(word.audio_data,word.sample_rate,pitch_range=[60, 400], trendline_quefrency_range=[0.0001, 0.05], cepstrum = 'real_cepstrum',plotting = True)
    #get_cppplots_per_speaker_and_disorder(words_segments)