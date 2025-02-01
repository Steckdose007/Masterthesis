import numpy as np
import librosa
import scipy.fft
import matplotlib.pyplot as plt
from audiodataloader import AudioDataLoader, AudioSegment
import train_gmm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

ubm = None
# Apply Hamming window and frame the signal
def frame_signal(signal, frame_size, hop_size):
    frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_size, axis=0)
    window = np.hamming(frame_size)
    return frames * window

# Compute the spectral envelope (using a 2048-point DFT and cepstral smoothing)
def compute_spectral_envelope(frames, sample_rate, n_fft=2048, cepstral_order=60):
    spectral_envelopes = []
    for frame in frames:
        # Apply FFT
        spectrum = np.abs(scipy.fft.fft(frame, n=n_fft))[:n_fft // 2]
        # Apply logarithm to the spectrum (cepstral step)
        log_spectrum = np.log(spectrum + 1e-10)  # Adding a small constant to avoid log(0)
        # Apply the inverse FFT to get the cepstrum
        cepstrum = np.real(scipy.fft.ifft(log_spectrum))
        # Apply cepstral smoothing by keeping only the first `cepstral_order` coefficients
        smoothed_spectrum = scipy.fft.fft(cepstrum[:cepstral_order], n=n_fft)
        spectral_envelopes.append(np.abs(smoothed_spectrum[:n_fft // 2]))
    return np.array(spectral_envelopes)

# Compute energy in specific frequency bands (e.g., 5-11 kHz, 11-20 kHz)
def compute_energy_in_bands(spectral_envelopes, sample_rate, bands=[(5000, 11000), (11000, 20000)]):
    freqs = np.fft.fftfreq(spectral_envelopes.shape[1], 1/sample_rate)[:spectral_envelopes.shape[1]]
    energy_bands = np.zeros((spectral_envelopes.shape[0], len(bands)))
    
    for i, band in enumerate(bands):
        band_indices = np.where((freqs >= band[0]) & (freqs < band[1]))[0]
        energy_bands[:, i] = np.sum(spectral_envelopes[:, band_indices] ** 2, axis=1)
    
    return np.mean(energy_bands, axis=0)  # Average over all frames

# Compute MFCCs, Delta MFCCs, and Delta-Delta MFCCs
def compute_mfcc_features(signal, sample_rate, n_mfcc=12, n_mels=22, frame_size=25.6e-3, hop_size=10e-3, n_fft=2048):
    # Convert frame and hop size from seconds to samples
    frame_length = int(frame_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    
    # Compute the static MFCCs using librosa's mfcc function
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, 
                                 n_fft=n_fft, hop_length=hop_length, win_length=frame_length, n_mels=n_mels)
    
    # Compute the first-order difference (Delta MFCCs) using a 5-frame window
    mfcc_delta = librosa.feature.delta(mfccs, width=5)
    
    # Compute the second-order difference (Delta-Delta MFCCs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=5)
    
    # Concatenate static, delta, and delta-delta features to form a 36-dimensional feature vector per frame
    mfcc_features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
    
    return mfcc_features

def plot_mel_spectrogram(word, phone=None):
    signal = word.audio_data
    sample_rate = word.sample_rate

    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128)

    # Convert power spectrogram to decibel (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel-spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(label='Decibels (dB)')
    plt.title(f'Mel-Spectrogram for {word.label}')

    if phone:
        # Adjust phone start and end times relative to the word start (assuming start and end times are in sample indices)
        phone_start_time = (phone.start_time - word.start_time) / sample_rate
        phone_end_time = (phone.end_time - word.start_time) / sample_rate
        # Plot vertical lines for phone start and end times
        plt.axvline(x=phone_start_time, color='green', linestyle='--', label='Phone Start')
        plt.axvline(x=phone_end_time, color='red', linestyle='--', label='Phone End')

        plt.legend()

    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency (Hz)')
    plt.show()

# Plot spectrogram with frequencies on the y-axis and time on the x-axis
def plot_spectrogram(signal, sample_rate, label, n_fft=2048, hop_length=512):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Convert the amplitude spectrogram to dB-scaled spectrogram (log scale)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram (dB) for {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

def plot_frequencies(spectral_envelopes):#, spectral_envelopes1):
    # Convert amplitude to dB (logarithmic scale) for both sets of spectral envelopes
    spectral_envelopes_db = 20 * np.log10(spectral_envelopes + 1e-10)  # Adding small constant to avoid log(0)
    
    # Calculate frequency bins
    frequencies = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2]

    # Plot the first set of spectral envelopes in dB
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.mean(spectral_envelopes_db, axis=0), label='Spectral Envelopes interdental', color='blue')

    # Plot the second set of spectral envelopes in dB
    #spectral_envelopes1_db = 20 * np.log10(spectral_envelopes1 + 1e-10)  # Adding small constant to avoid log(0)
    #plt.plot(frequencies, np.mean(spectral_envelopes1_db, axis=0), label='Spectral Envelopes normal', color='red')

    # Add titles and labels
    plt.title(f'Spectral Envelopes Comparison for {word.label}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.show()


def pad_mfccs(mfccs_train,mfccs_test):
    """
    Pad MFCC sequences to the same length as the longest MFCC sequence.
    
    Parameters:
    - mfccs: List of MFCC sequences (each is a 2D array with shape (n_frames, n_features)).
    - max_length: The length to pad each MFCC sequence to (equal to the longest MFCC sequence).
    
    Returns:
    - padded_mfccs: List of padded MFCC sequences with shape (max_length, n_features).
    """
    max_length = max(max([mfcc.shape[0] for mfcc in mfccs_train]),max([mfcc.shape[0] for mfcc in mfccs_test])) #maximum of all mfccs
    print("maxlen: ",max_length)
    padded_mfccs_train = []
    for mfcc in mfccs_train:
        n_frames, n_features = mfcc.shape
        if n_frames < max_length:
            # Pad with zeros if the sequence is shorter than the max length
            padding = np.zeros((max_length - n_frames, n_features))
            padded_mfcc = np.vstack((mfcc, padding))
        else:
            # No truncation needed since we want to match the longest MFCC
            padded_mfcc = mfcc
        padded_mfccs_train.append(padded_mfcc.flatten())
    padded_mfccs_test = []
    for mfcc in mfccs_test:
        n_frames, n_features = mfcc.shape
        if n_frames < max_length:
            # Pad with zeros if the sequence is shorter than the max length
            padding = np.zeros((max_length - n_frames, n_features))
            padded_mfcc = np.vstack((mfcc, padding))
        else:
            # No truncation needed since we want to match the longest MFCC
            padded_mfcc = mfcc
        padded_mfccs_test.append(padded_mfcc.flatten())
    return padded_mfccs_train,padded_mfccs_test


def get_features(dataclass, energy_bool = False,training=False,reg_covar=1e-6,relevance_factor=15):
    global ubm
    mfcc_list = [] #here are all mfccs stored after building it with shape((n_components, n_features))
    for segment in dataclass:
        signal = segment.audio_data
        sample_rate = segment.sample_rate
        #print(segment.label,segment.label_path)
        # Compute 12 static MFCCs, 24 dynamic (delta and delta-delta) MFCCs, using 22 Mel filters
        mfcc = train_gmm.compute_mfcc_features(signal, sample_rate)
        #Transpose to get it like that: (n_components, n_features) for the covarianve_type: diag
        mfcc_list.append(np.transpose(mfcc))
    
    if(training):
        # Concatenate all MFCC features into a single matrix So (n_frames,36 features) for ubm training
        mfcc_features = np.concatenate(mfcc_list, axis=0)
        # scaler = StandardScaler()
        # mfcc_features = scaler.fit_transform(mfcc_features)
        print("Training UBM...")
        ubm = train_gmm.train_ubm(mfcc_features, n_components=16, max_iter=100, reg_covar=reg_covar)#safed in a public variable so it can be used when test is called an no training on test data
        print("Training finished!")

    # Step 2: Adapt the UBM for each word
    labels = []
    supervectors = []
    energys = []
    simmplified_supervectors = []
    for n,segment in enumerate(dataclass):
        signal = segment.audio_data
        mfcc = mfcc_list[n] #has shape (n_components, 36)
        # scaler = StandardScaler()
        # mfcc = scaler.fit_transform(mfcc)
        #print("mfcc word adaption shape:", np.shape(mfcc))
        if(ubm == None):
            print("Error! First call Train to init UBM")
            return None
        adapted_gmm = train_gmm.adapt_ubm_map(ubm, mfcc, relevance_factor=relevance_factor)
        
        # Step 3: Extract the supervector
        supervector, simmplified_supervector = train_gmm.extract_supervector(adapted_gmm)
        #print(np.shape(supervector),np.shape(simmplified_supervector))
        supervectors.append(supervector)
        simmplified_supervectors.append(simmplified_supervector)
        labels.append(segment.label_path)
        if energy_bool:
            # Frame the signal and apply Hamming window
            frame_size = int(25.6e-3 * sample_rate)  # Frame size (25.6 ms)
            hop_size = int(10e-3 * sample_rate)      # Frame shift (10 ms)
            frames = frame_signal(signal, frame_size, hop_size)
            # Compute the spectral envelope for each frame
            n_fft=2048
            cepstral_order=60
            spectral_envelopes = compute_spectral_envelope(frames, sample_rate, n_fft, cepstral_order)
            # Compute energy in specific frequency bands (5-11 kHz and 11-20 kHz)
            energy_features = compute_energy_in_bands(spectral_envelopes, sample_rate)
            energys.append(energy_features)
    #pad to have same lenght.
    #mfccs per word are also flattened

    if energy_bool:
            print(f"Extracted {len(supervectors)} supervectors, {len(simmplified_supervectors)} Simplified Supervectors,{len(mfcc_list) } MFCCs and {len(energys)} Energy.")
            return supervectors,simmplified_supervectors,mfcc_list,energys, labels
    
    print(f"Extracted {len(supervectors)} supervectors, {len(simmplified_supervectors)} simplified supervectors,{len(mfcc_list) } MFCCs.")
    return supervectors,simmplified_supervectors,mfcc_list, labels

def grid_search(dataclass, reg_covar_values, relevance_factor_values, energy_bool=False):
    best_score = 0
    best_params = {'reg_covar': None, 'relevance_factor': None}
    all_results = []
    global ubm

    segments_train, segments_test = train_test_split(words_segments, random_state=42,test_size=0.20)
    # Loop over all combinations of reg_covar and relevance_factor
    for reg_covar in reg_covar_values:
        for relevance_factor in relevance_factor_values:
            print(f"Testing reg_covar={reg_covar}, relevance_factor={relevance_factor}")
            

            supervectors_train, simplified_supervectors_train, mfccs_train, energys_train, labels_train = get_features(segments_train,energy_bool=True,training=True,reg_covar=reg_covar,relevance_factor=relevance_factor)#has to be called first to init ubm
            supervectors_test, simplified_supervectors_test, mfccs_test, energys_test, labels_test = get_features(segments_test,energy_bool=True,reg_covar=reg_covar,relevance_factor=relevance_factor)

            scaler = StandardScaler()
            supervectors_train = scaler.fit_transform(supervectors_train)
            supervectors_test = scaler.transform(supervectors_test)
            #simplified_supervectors= scaler.fit_transform(simplified_supervectors)
            #mfccs= scaler.fit_transform(mfccs)
            
            label_encoder = LabelEncoder()
            encoded_labels_train = label_encoder.fit_transform(labels_train)
            encoded_labels_test = label_encoder.transform(labels_test)

            X_train =supervectors_train
            X_test =supervectors_test

            # Create and train the SVM with a polynomial kernel
            svm_classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
            svm_classifier.fit(X_train, encoded_labels_train)
            y_pred = svm_classifier.predict(X_test)
            #get predicted probabilities
            svm_prob = SVC(kernel='poly', degree=3, C=1.0, probability=True)
            svm_prob.fit(X_train, encoded_labels_train)
            y_pred_proba = svm_prob.predict_proba(X_test)

            # Compute the metrics
            metrics = compute_metrics(encoded_labels_test, y_pred, y_pred_proba)
            all_results.append((reg_covar, relevance_factor, metrics))
            print(metrics)
            # Update best parameters if current score is better
            current_score = metrics['AUC']  # Use AUC as the metric to optimize
            if current_score > best_score:
                best_score = current_score
                best_params = {'reg_covar': reg_covar, 'relevance_factor': relevance_factor}
                print(f"New best score: {best_score} with reg_covar={reg_covar}, relevance_factor={relevance_factor}")
            ubm = None
    return best_params, best_score, all_results


def display_results_table(all_results, metric_name='AUC'):
    """
    Displays the results from the grid search in a table format.
    
    Parameters:
    - all_results: A list of tuples (reg_covar, relevance_factor, metrics_dict)
    - metric_name: The name of the metric to display (default is 'AUC')
    """
    # Extract unique reg_covar and relevance_factor values
    reg_covar_values = sorted(list(set([result[0] for result in all_results])))
    relevance_factor_values = sorted(list(set([result[1] for result in all_results])))

    # Create a DataFrame to hold the metric values
    metric_df = pd.DataFrame(index=reg_covar_values, columns=relevance_factor_values)

    # Fill the DataFrame with the corresponding metric values
    for result in all_results:
        reg_covar = result[0]
        relevance_factor = result[1]
        metric_value = result[2][metric_name]
        metric_df.loc[reg_covar, relevance_factor] = round(metric_value, 2)

    # Display the DataFrame as a formatted table
    print(f"\nTable of {metric_name} for Different reg_covar and relevance_factor Combinations:\n")
    print(metric_df)


def compute_metrics(y_true, y_pred, y_pred_proba):
    
    # Recognition Rate (RR) = Accuracy
    RR = accuracy_score(y_true, y_pred)
    # Recognition Rate for Normal class (Rn) and Pathological class (Rp)
    Rn = recall_score(y_true, y_pred, pos_label=0)  # Recall for class 0 (Normal class)
    Rp = recall_score(y_true, y_pred, pos_label=1)  # Recall for class 1 (Pathological class)
    # Class-wise Averaged Recognition Rate (CL)
    CL = (Rn + Rp) / 2   
    # Area Under the Curve (AUC)
    AUC = roc_auc_score(y_true, y_pred_proba[:, 1])  # Use probabilities for positive class   
    return {
        'RR': float(round(RR, 3)),
        'Rn': float(round(Rn, 3)),
        'Rp': float(round(Rp, 3)),
        'CL': float(round(CL, 3)),
        'AUC': float(round(AUC, 3))
    }


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

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=False)
    #phones_segments = loader.load_segments_from_pickle("phones_segments.pkl")
    words_segments = loader.load_segments_from_pickle("data_lists\words_normalized_44kHz.pkl")
    #sentences_segments = loader.load_segments_from_pickle("sentences_segments.pkl")

    """get search for best hyperparam"""
    # # Define the parameter search space
    # reg_covar_values = [1e-6, 1e-5, 1e-4, 1e-3]
    # relevance_factor_values = [10, 15, 20, 25, 30]

    # # Call the grid search function
    # best_params, best_score, all_results = grid_search(words_segments, reg_covar_values, relevance_factor_values, energy_bool=True)

    # # Plot the results
    # print(f"Best parameters: {best_params} with score: {best_score}")
    # display_results_table(all_results, metric_name='AUC')



    segments_train, segments_val, segments_test= split_list_after_speaker(words_segments)

    print(np.shape(segments_train),np.shape(segments_val))

    supervectors_train, simplified_supervectors_train, mfccs_train, energys_train, labels_train = get_features(segments_train,energy_bool=True,training=True)#has to be called first to init ubm
    supervectors_test, simplified_supervectors_test, mfccs_test, energys_test, labels_test = get_features(segments_val,energy_bool=True)
    padded_mfccs_train,padded_mfccs_test = pad_mfccs(mfccs_train,mfccs_test)#shape(396,23904)

    scaler = StandardScaler()
    supervectors_train = scaler.fit_transform(supervectors_train)
    supervectors_test = scaler.transform(supervectors_test)
    #simplified_supervectors= scaler.fit_transform(simplified_supervectors)
    #mfccs= scaler.fit_transform(mfccs)
    
    label_encoder = LabelEncoder()
    encoded_labels_train = label_encoder.fit_transform(labels_train)
    encoded_labels_test = label_encoder.transform(labels_test)

    print(np.shape(supervectors_train),np.shape(simplified_supervectors_train),np.shape(padded_mfccs_train),np.shape(energys_train),np.shape(labels_train))
    print(np.shape(supervectors_test),np.shape(simplified_supervectors_test),np.shape(padded_mfccs_test),np.shape(energys_test),np.shape(labels_test))
   
    X = [supervectors_train, simplified_supervectors_train, padded_mfccs_train, energys_train, encoded_labels_train]
    Y = [supervectors_test, simplified_supervectors_test, padded_mfccs_test, energys_test, encoded_labels_test]
    descriptions = ['Supervectors', 'SimplifiedSupervectors', 'MFCCs', 'Energy']
    m = []
    """SVM"""
    # for i in range(4):
    #     X_train = X[i]
    #     X_test =Y[i]
    #     y_train =X[4]
    #     y_test  =Y[4]
    #     # Create and train the SVM with a polynomial kernel
    #     svm_classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
    #     svm_classifier.fit(X_train, y_train)
    #     y_pred = svm_classifier.predict(X_test)
    #     #get predicted probabilities
    #     svm_prob = SVC(kernel='poly', degree=3, C=1.0, probability=True)
    #     svm_prob.fit(X_train, y_train)
    #     y_pred_proba = svm_prob.predict_proba(X_test)

    #     # Compute the metrics
    #     metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    #     print(metrics)
    #     m.append(metrics)
    """ADABOOSTM1"""
    for i in range(4):
        # Split the data into training and testing sets
        X_train = X[i]
        X_test =Y[i]
        y_train =X[4]
        y_test  =Y[4]

        # Create and train the AdaBoost classifier with DecisionTree as the base estimator
        base_estimator = DecisionTreeClassifier(max_depth=1)
        ada_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
        ada_classifier.fit(X_train, y_train)
        
        # Predict the class labels
        y_pred = ada_classifier.predict(X_test)
        
        # Get predicted probabilities for positive class
        y_pred_proba = ada_classifier.predict_proba(X_test)

        # Compute the metrics
        metrics = compute_metrics(y_test, y_pred, y_pred_proba)
        print(f"Metrics for {descriptions[i]}: {metrics}")
        m.append(metrics)


    df_metrics = pd.DataFrame(m)
    df_metrics.insert(0, 'Description', descriptions)
    # Convert the DataFrame to LaTeX format
    latex_table = df_metrics.to_latex(index=False,float_format="{:.2f}".format)
    print(latex_table)