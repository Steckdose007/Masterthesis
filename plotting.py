
import numpy as np
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.signal import savgol_filter
def plot_mel_spectrogram(word, phones=None):
    print("start: ",word.start_time/44100,phones[0].start_time/44100)
    print("end: ",word.end_time/44100,phones[0].end_time/44100)

    signal = word.audio_data
    sample_rate = 24000
    label = word.label_path
    scaling=24000/44100
    # Compute Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})

    # Plot the audio waveform
    axs[0].plot(signal, color='gray')
    axs[0].set_title(f"Audio Waveform for {word.label}  {word.label_path}")
    axs[0].set_xlabel("Time (samples)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()

    # Plot phone boundaries on the waveform
    if phones:
        for p in phones:
            # Adjust phone start and end times relative to the word start
            phone_start_sample = abs(int((p.start_time*scaling - word.start_time*scaling)))
            phone_end_sample = abs(int((p.end_time*scaling - word.start_time*scaling)))
            
            # Plot vertical lines for phone start and end times
            axs[0].axvline(x=phone_start_sample, color='green', linestyle='--', label='Phone Start')
            axs[0].axvline(x=phone_end_sample, color='red', linestyle='--', label='Phone End')
            

    axs[0].legend()

    # Plot the Mel spectrogram
    librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', cmap='coolwarm', ax=axs[1])
    axs[1].set_title(f"Mel Spectrogram for {word.label} {word.label_path}")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Mel Frequency (Hz)")

    # Plot phone boundaries on the spectrogram
    if phones:
        for p in phones:
            # Adjust phone start and end times relative to the word start (in seconds)
            phone_start_time = abs((p.start_time*scaling - word.start_time*scaling)) / sample_rate
            phone_end_time = abs((p.end_time*scaling - word.start_time*scaling)) / sample_rate

            # Plot vertical lines for phone start and end times
            axs[1].axvline(x=phone_start_time, color='green', linestyle='--', label='Phone Start')
            axs[1].axvline(x=phone_end_time, color='red', linestyle='--', label='Phone End')
            

    axs[1].legend()

    # Add a color bar to the spectrogram
    #cbar = fig.colorbar(axs[1].images[0], ax=axs[1], format='%+2.0f dB')
    #cbar.set_label("Decibels (dB)")

    # Adjust layout
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



