"""
Reamples the Data folder to any new sample rate
"""
import librosa
import soundfile as sf
import os

def resample_audio_folder(input_dir, output_dir, target_sr=16000):
    """Resample all .wav files in input_dir to target_sr and save in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load at original rate
            audio, sr = librosa.load(input_path, sr=None)  # sr=None keeps the original rate

            # Resample only if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Save the resampled audio
            sf.write(output_path, audio, target_sr)

# --- Main part of the script ---

base_dir = "data"  # The folder containing 'normal' and 'sigmatism'
normal_input = os.path.join(base_dir, "normal")
sigmatism_input = os.path.join(base_dir, "sigmatism")

normal_output = os.path.join(base_dir, "normal16kHz")
sigmatism_output = os.path.join(base_dir, "sigmatism16kHz")

# Resample both folders
resample_audio_folder(normal_input, normal_output, target_sr=16000)
resample_audio_folder(sigmatism_input, sigmatism_output, target_sr=16000)

print("Resampling complete! Check normal16kHz and sigmatism16kHz folders for results.")