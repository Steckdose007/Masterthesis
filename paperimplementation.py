import numpy as np
import librosa
import scipy.fft
import matplotlib.pyplot as plt
from audiodataloader import AudioDataLoader



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


if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= True, phone_data= True, sentence_data= True)    
    words_segments = loader.create_dataclass_words()
    phone_segments = loader.create_dataclass_phones()
    phone = phone_segments[82]
    word = words_segments[36]
    signal = word.audio_data
    sample_rate = word.sample_rate

 
    # Frame the signal and apply Hamming window
    frame_size = int(25.6e-3 * sample_rate)  # Frame size (25.6 ms)
    hop_size = int(10e-3 * sample_rate)      # Frame shift (10 ms)
    print(f"Sample Rate for word {word.label}: ",sample_rate)
    frames = frame_signal(signal, frame_size, hop_size)
    print(f"{word.label} frames: ", np.shape(frames)) #(43, 1128) i have 43 frames each with 1128 samples.(the number of time-domain samples in each frame)
    # Compute the spectral envelope for each frame
    n_fft=2048
    cepstral_order=60
    spectral_envelopes = compute_spectral_envelope(frames, sample_rate, n_fft, cepstral_order)
    print(f"{word.label} spectral_envelopes: ", np.shape(spectral_envelopes))#(43, 1024) for each of the 43 frames there are 1024 frequency-domain values. 
    

    #Some plotting
    #plot_spectrogram(signal, sample_rate, word.label)
    plot_frequencies(spectral_envelopes)#,spectral_envelopes1)
    #plot_mel_spectrogram(word,phone)


    # Compute energy in specific frequency bands (5-11 kHz and 11-20 kHz)
    energy_features = compute_energy_in_bands(spectral_envelopes, sample_rate)
    print(f"Energy in 5-11 kHz: {energy_features[0]}, Energy in 11-20 kHz: {energy_features[1]}")


    # Compute 12 static MFCCs, 24 dynamic (delta and delta-delta) MFCCs, using 22 Mel filters
    mfcc_features = compute_mfcc_features(signal, sample_rate)
    # mfcc_features will have the shape (36, num_frames), where 36 is the number of MFCC coefficients (12 static, 24 dynamic)
    print(f'MFCC features shape: {mfcc_features.shape}')

