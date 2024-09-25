import numpy as np
import librosa
import scipy.fft

# Load the audio file
def load_audio(file_path, sr=44100):
    # Use librosa to load the audio file and resample to 44.1 kHz
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate

# Extract energy from specific frequency bands
def compute_energy(signal, sample_rate, bands=[(5000, 11000), (11000, 20000)]):
    # Number of samples in the audio
    N = len(signal)
    
    # Apply Fourier Transform
    fft_result = scipy.fft.fft(signal)
    
    # Get the corresponding frequencies
    freqs = scipy.fft.fftfreq(N, 1/sample_rate)
    
    # Only take the positive half of the spectrum
    positive_freqs = freqs[:N//2]
    positive_fft = np.abs(fft_result[:N//2])
    
    # Initialize energy list
    energy_bands = []
    
    # For each frequency band, compute the energy
    for band in bands:
        # Find indices for the current band
        band_indices = np.where((positive_freqs >= band[0]) & (positive_freqs < band[1]))[0]
        # Compute the energy as the sum of squared magnitudes in this band
        energy = np.sum(positive_fft[band_indices] ** 2)
        energy_bands.append(energy)
    
    return energy_bands

# Example usage
file_path = 'path_to_your_audio.wav'  # Replace with your .wav file path
signal, sample_rate = load_audio(file_path)

# Extract energy in 5-11 kHz and 11-20 kHz bands
energy_features = compute_energy(signal, sample_rate)
print(f"Energy in 5-11 kHz: {energy_features[0]}, Energy in 11-20 kHz: {energy_features[1]}")
