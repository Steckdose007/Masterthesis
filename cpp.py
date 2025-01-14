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
from Dataloader_pytorch import AudioSegmentDataset ,process_and_save_dataset
import pandas as pd
from scipy import linalg
import scipy.stats as stats
import warnings
from tqdm import tqdm 
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
    """CPP for normal and sigmatism over all words"""
    sigmatism = []
    normal = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        #print(filename1)
        extracted = extract_phone_segments(word)
        cpp_peak = cpp_calc_and_plot(extracted,word.sample_rate,pitch_range=[60, 1000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
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
        cpp_peak = cpp_calc_and_plot(extracted,word.sample_rate,pitch_range=[60, 1000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
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

def paired_t_test(word_segemtns):
    """CPP per word per speaker"""
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        if(word.label_path == "sigmatism"):
            filename1 = filename1.replace("_sig", "")
        #extracted = extract_phone_segments(word)
        cpp_peak = cpp_calc_and_plot(word.audio_data,word.sample_rate,pitch_range=[60, 1000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = False)
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
    data = []
    for word in words_segments:
        filename1 = os.path.splitext(os.path.basename(word.path))[0]  
        if(word.label_path == "sigmatism"):
            filename1 = filename1.replace("_sig", "")
        audio = fix_audio_length(word.audio_data,int(1.2*24000))
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
    plt.boxplot(df_clean['FID'])
    plt.ylabel("FID (normal vs. sigmatism)")
    plt.title("Distribution of FID Scores Across All Speakers")
    plt.show()

    speaker_means_df = df_clean.groupby('Speaker')['FID'].mean().reset_index()
    speaker_means_df.columns = ['Speaker', 'MeanFID']

    # Make a new boxplot
    plt.figure(figsize=(6, 6))
    plt.boxplot(speaker_means_df['MeanFID'])
    plt.ylabel("Mean FID (across words)")
    plt.title("Distribution of Mean FIDs Per Speaker")
    plt.show()

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


if __name__ == "__main__":
    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True, downsample=True)
    phones_segments = loader.load_segments_from_pickle("phones__24kHz.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    mfcc_dim={
        "n_mfcc":128, 
        "n_mels":128, 
        "frame_size":0.025, 
        "hop_size":0.005, 
        "n_fft":2048,
        "target_length": 224
    }
    segments = AudioSegmentDataset(words_segments,phones_segments, mfcc_dim, augment= False)
    mu1, sigma1 = compute_mean_and_cov(words_segments[0].audio_data)
    mu2, sigma2 = compute_mean_and_cov(words_segments[0].audio_data)
    eps=1e-6
    fid_value = frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
    print(fid_value)
    FAD_libary()
    #fid_plotting(words_segments)
    #paired_t_test(words_segments)
    #word = words_segments[0]
    #cpp_calc_and_plot(word.audio_data,word.sample_rate,pitch_range=[60, 1000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = True)
    #get_cppplots_per_speaker_and_disorder(words_segments)