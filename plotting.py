
import numpy as np
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.signal import savgol_filter
import data_augmentation

def plot_mel_spectrogram(word, phones=None):

    signal = word.audio_data
    sample_rate = word.sample_rate
    scaling=sample_rate/44100
    # Compute Mel-spectrogram
    n_mels=128
    frame_size=0.025
    hop_size=0.005
    n_mfcc=128
    n_fft=2048
    frame_length = int(frame_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=frame_length
    )
    mfccs = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, n_mels=n_mels
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    print(np.shape(mel_spectrogram_db))
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})

    # Plot the audio waveform
    axs[0].plot(signal, color='gray')
    axs[0].set_title(f"Audio Waveform for {word.label}  {word.label_path}")
    axs[0].set_xlabel("Time (samples)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()
    
    """Approach 2 see twist"""
    # # Normalize word length into equal parts
    # num_chars = len(word.label)
    # signal_per_char = signal.shape[0] // num_chars
    # phone_chars=['s','S','Z', 'z','X', 'x']
    # # Find positions of the phone characters in the word
    # phone_positions = [i for i, char in enumerate(word.label) if char in phone_chars]
    # for position in phone_positions:
    #     char_start_signal = position * signal_per_char
    #     char_end_signal = (position + 1) * signal_per_char
    #     axs[0].axvline(x=char_start_signal, color='green', linestyle='--', label='Phone Start')
    #     axs[0].axvline(x=char_end_signal, color='red', linestyle='--', label='Phone End')

    
    """APProach 1 see twist"""    
    #Plot phone boundaries on the waveform
    if phones:
        for p in phones:
            # Adjust phone start and end times relative to the word start
            phone_start_sample = abs(int((p.start_time - word.start_time)*scaling))
            phone_end_sample = abs(int((p.end_time - word.start_time)*scaling))
            print("start: ",phone_start_sample)
            print("end: ",phone_end_sample)
            
            # Plot vertical lines for phone start and end times
            axs[0].axvline(x=phone_start_sample, color='green', linestyle='--', label='Phone Start')
            axs[0].axvline(x=phone_end_sample, color='red', linestyle='--', label='Phone End')
            

    axs[0].legend()

    # Plot the Mel spectrogram
    img = axs[1].imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='plasma')
    cbar = plt.colorbar(img, ax=axs[1], orientation='vertical', pad=0.01)
    cbar.set_label('Intensity (dB)', rotation=270, labelpad=15)
    #librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', cmap='coolwarm', ax=axs[1])
    axs[1].set_title(f"Mel Specto for {word.label} {word.label_path}")
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("Coefficients")
    
    # """Approach 2 see twist"""
    # # Normalize word length into equal parts
    # num_chars = len(word.label)
    # frames_per_char = mel_spectrogram_db.shape[1] // num_chars
    # phone_chars=['s','S','Z', 'z','X', 'x']
    # # Find positions of the phone characters in the word
    # phone_positions = [i for i, char in enumerate(word.label) if char in phone_chars]
    # for position in phone_positions:
    #     char_start_frame = position * frames_per_char
    #     char_end_frame = (position + 1) * frames_per_char
    #     axs[1].axvline(x=char_start_frame, color='green', linestyle='--', label='Phone Start')
    #     axs[1].axvline(x=char_end_frame, color='red', linestyle='--', label='Phone End')

    """APProach 1 see twist"""    
    hop_length = int(0.005 * sample_rate)
    #Plot phone boundaries on the spectrogram
    if phones:
        for p in phones:
            # Adjust phone start and end times relative to the word start (in seconds)
            frame_start = abs(int(((p.start_time - word.start_time)*scaling) / hop_length))
            frame_end = abs(int(((p.end_time - word.start_time)*scaling) / hop_length))
            print("start: ",frame_start)
            print("end: ",frame_end)
            # Plot vertical lines for phone start and end times
            axs[1].axvline(x=frame_start, color='green', linestyle='--', label='Phone Start')
            axs[1].axvline(x=frame_end, color='red', linestyle='--', label='Phone End')
            

    axs[1].legend()

    # Add a color bar to the spectrogram
    #cbar = fig.colorbar(axs[1].images[0], ax=axs[1], format='%+2.0f dB')
    #cbar.set_label("Decibels (dB)")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_mfcc_and_mel_spectrogram(segment, sample_rate=24000, n_mfcc=128, n_mels=128, frame_size=0.025, hop_size=0.005, n_fft=2048):
    """
    Plot MFCCs and Mel Spectrogram side by side.
    
    Parameters:
    - signal: The audio signal (1D NumPy array).
    - sample_rate: Sampling rate of the audio signal.
    - n_mfcc: Number of MFCC coefficients to compute.
    - n_mels: Number of Mel bands to use for the spectrogram.
    - frame_size: Frame size in seconds.
    - hop_size: Hop size in seconds.
    - n_fft: Number of FFT points.
    """
    signal = segment.audio_data
    sample_rate = segment.sample_rate
    # Convert frame and hop sizes to samples
    frame_length = int(frame_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    
    # Compute Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=frame_length
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, n_mels=n_mels
    )
    print(np.shape(mel_spectrogram))
    print(np.shape(mfccs))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mel Spectrogram
    img1 = librosa.display.specshow(
        mel_spectrogram_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[0], cmap='coolwarm'
    )
    axes[0].set_title('Mel Spectrogram')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # MFCCs
    img2 = librosa.display.specshow(
        mfccs, sr=sample_rate, hop_length=hop_length, x_axis='time', ax=axes[1], cmap='coolwarm'
    )
    axes[1].set_title('MFCCs')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('MFCC Coefficients')
    fig.colorbar(img2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def compare_spectral_envelopes(word1, word2, n_fft=2048, smoothing_window=51, poly_order=3):
    
    # Extract audio data and sample rates
    signal1, sr1, label1 = word1.audio_data, word1.sample_rate, word1.label_path
    signal2, sr2, label2 = word2.audio_data, word2.sample_rate, word2.label_path
    
    if sr1 != sr2:
        raise ValueError("Sample rates of the two words must be the same for comparison.")
    
    # Compute FFT and magnitude for the first word
    fft1 = np.fft.fft(signal1, n=n_fft)
    magnitude1 = np.abs(fft1[:n_fft // 2])  # Take positive frequencies only
    frequencies = np.fft.fftfreq(n_fft, 1 / sr1)[:n_fft // 2]
    spectral_envelope1 = savgol_filter(magnitude1, smoothing_window, poly_order)

    # Compute FFT and magnitude for the second word
    fft2 = np.fft.fft(signal2, n=n_fft)
    magnitude2 = np.abs(fft2[:n_fft // 2])  # Take positive frequencies only
    spectral_envelope2 = savgol_filter(magnitude2, smoothing_window, poly_order)

    # Plot the spectral envelopes
    plt.figure(figsize=(12, 8))
    plt.plot(frequencies, spectral_envelope1, label=f'Spectral Envelope ({label1})', color='blue', linewidth=2)
    plt.plot(frequencies, spectral_envelope2, label=f'Spectral Envelope ({label2})', color='orange', linewidth=2)
    plt.title(f'Comparison of Spectral Envelopes for {word1.label}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def visualize_augmentations(audio_data, sample_rate):
    """
    Visualize the effects of data augmentation on audio data.
    
    Parameters:
    - audio_data: The original audio signal (numpy array).
    - sample_rate: The sample rate of the audio signal.
    """
    

    # Target length for cropping/padding
    target_length = int(0.8 * len(audio_data))  # 80% of original length

    # Apply augmentations
    augmentations = [
        ("Original", audio_data),
        ("Gaussian Noise", data_augmentation.add_gaussian_noise(audio_data,sample_rate)),
        ("Time Stretch (slower)", data_augmentation.time_stretch(audio_data, sample_rate)),
        ("Pitch Shift (+2 semitones)", data_augmentation.pitch_shift(audio_data, sample_rate)),
        ("Random Crop/Pad", data_augmentation.random_crop_pad(audio_data, sample_rate))
    ]

    # Plot the augmentations
    plt.figure(figsize=(12, 8))
    for i, (title, augmented_data) in enumerate(augmentations):
        plt.subplot(len(augmentations), 1, i + 1)
        plt.plot(augmented_data, alpha=0.7, label=title)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend(loc='upper right')
        plt.grid()

    plt.tight_layout()
    plt.show()

