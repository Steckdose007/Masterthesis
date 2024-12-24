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
import pandas as pd


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


if __name__ == "__main__":
    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True, downsample=True)
    
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    word = words_segments[0]
    cpp_calc_and_plot(word.audio_data,word.sample_rate,pitch_range=[60, 1000], trendline_quefrency_range=[0.001, 0.05], cepstrum = 'real_cepstrum',plotting = True)
    get_cppplots_per_speaker_and_disorder(words_segments)