
import numpy as np
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft

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

def plot_frequencies(spectral_envelopes):#, spectral_envelopes1):
    # Convert amplitude to dB (logarithmic scale) for both sets of spectral envelopes
    spectral_envelopes_db = 20 * np.log10(spectral_envelopes + 1e-10)  # Adding small constant to avoid log(0)
    n_fft=2048,
    sample_rate = 24000
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

