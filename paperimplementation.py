import numpy as np
import librosa
import scipy.fft
import matplotlib.pyplot as plt

# Load the audio file
def load_audio(file_path, sr=44100):
    # Use librosa to load the audio file and resample to 44.1 kHz
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate

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


# Plot spectrogram with frequencies on the y-axis and time on the x-axis
def plot_spectrogram(signal, sample_rate, n_fft=2048, hop_length=512):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Convert the amplitude spectrogram to dB-scaled spectrogram (log scale)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = 'Data\Speaker1gtSamen.wav'  
    #file_path = 'Data\Speaker1gtSamen.wav'  
    signal, sample_rate = load_audio(file_path)
    frame_size = int(25.6e-3 * sample_rate)  # Frame size (25.6 ms)
    hop_size = int(10e-3 * sample_rate)      # Frame shift (10 ms)
    print("Sample Rate: ",sample_rate)
    # Frame the signal and apply Hamming window
    frames = frame_signal(signal, frame_size, hop_size)
    print("frames: ", np.shape(frames)) #(43, 1128) i have 43 frames each with 1128 samples.(the number of time-domain samples in each frame)
    # Compute the spectral envelope for each frame
    n_fft=2048
    cepstral_order=60
    spectral_envelopes = compute_spectral_envelope(frames, sample_rate, n_fft, cepstral_order)
    print("spectral_envelopes: ", np.shape(spectral_envelopes))#(43, 1024) for each of the 43 frames there are 1024 frequency-domain values. 


    #Some plotting
    # Plot the spectrogram
    plot_spectrogram(signal, sample_rate)
    mean_spectral_envelope = np.mean(spectral_envelopes, axis=0)
    # Convert amplitude to dB (logarithmic scale)
    mean_spectral_envelope_db = 20 * np.log10(mean_spectral_envelope + 1e-10)  # Adding a small constant to avoid log(0)
    frequencies = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2]
    # Plot the mean spectral envelope in dB with actual frequency values
    plt.plot(frequencies, mean_spectral_envelope_db, label='Mean Spectral Envelope (dB)', color='blue')
    plt.title('Mean Spectral Envelope with Frequencies (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.legend()
    plt.show()


    # Compute energy in specific frequency bands (5-11 kHz and 11-20 kHz)
    energy_features = compute_energy_in_bands(spectral_envelopes, sample_rate)
    print(f"Energy in 5-11 kHz: {energy_features[0]}, Energy in 11-20 kHz: {energy_features[1]}")



    # Compute 12 static MFCCs, 24 dynamic (delta and delta-delta) MFCCs, using 22 Mel filters
    mfcc_features = compute_mfcc_features(signal, sample_rate)
    # mfcc_features will have the shape (36, num_frames), where 36 is the number of MFCC coefficients (12 static, 24 dynamic)
    print(f'MFCC features shape: {mfcc_features.shape}')

