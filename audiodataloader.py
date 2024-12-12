"""
Workflow Summary:

    Training Phase:
        Collect a large dataset of speech data (multiple speakers, various utterances).
        Extract MFCC features from the entire dataset.
        Train the UBM (GMM) using the MFCC features from this dataset.

    Testing (Adaptation) Phase:
        For each new word (which the UBM has not seen before):
            Extract MFCC features from the word.
            Adapt the UBM using these MFCC features to create a GMM specific to the word.
            Extract the supervector from the adapted GMM.
            Use the supervector as a feature for classification (e.g., using an SVM or another classifier).

"""
"""
The AudioDataLoader class is designed to load and process audio data along with its corresponding annotation 
(CSV file), extracting specific segments such as words, sentences, and phones (phonetic units) from the audio. 
The words are extracted by dividing them along their pauses. So there has to be a pause between them. 
Sentences are Hardcoded in the config and are divided by their own biginning and ending which has to be decided manual. 
Phones are automatically extracted from all words with ["z","s","Z","S","ts"].
Dividing_word is in the config the word which divedes the word segemtns from the sentences in the audio. EG. Xylophon
returns: a list with AudioSegment objects which either has phones,words,sentences. To get one of those set according parameter to True.
Get_Buffer sets a bool which puts a buffer at the end and beginning of the segments.

        

"""
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import json
import os
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np
import pandas as pd
from numba import jit
import timeit 
import plotting
sum_length =0

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  #word for example sonne
    label_path: str # sigmatism or normal
    path: str # which file it is from
    #phonem_loc : list

@jit(nopython=True)
def rolling_std(signal, window_size):
    """
    Compute the rolling standard deviation over a given window size.
    
    Parameters:
    - signal: The audio signal (1D numpy array).
    - window_size: The size of the window (in samples).
    
    Returns:
    - rolling_std: Rolling standard deviation of the signal.
    """
    return np.array([np.std(signal[i:i+window_size]) for i in range(len(signal) - window_size)])


class AudioDataLoader:
    def __init__(self, config_file: str, phone_data: bool = False, word_data: bool = False, sentence_data: bool = False, get_buffer: bool = False, downsample : bool = False):
        self.phone_bool = phone_data
        self.word_bool = word_data
        self.sentence_bool = sentence_data
        self.word_segments = []
        self.target_sr = 24000
        self.org_sample_rate = 44100
        self.sentence_segments = []
        self.phone_segments = []
        self.phones =["z","s","Z","S","ts"]
        self.downsample = downsample
        self.folder_path = None
        self.dividing_word = None
        self.maximum_word_length = 0
        self.label_path = None
        self.get_buffer = get_buffer
        self.buffer = 0.005 #5ms
        self.buffer_word = 0.065#window to search for word
        self.load_config(config_file)
        self.files = self.get_audio_csv_file_pairs()

    def load_config(self, config_file: str):
            """Load sentences and dividing word from the JSON config file."""
            with open(config_file, 'r') as file:
                config = json.load(file)
            
            self.sentences = config["sentences"]  
            self.dividing_word = config["dividing_word"]  # Load the dividing word (e.g., "Xylophone")
            self.folder_path = config["folder_path"]
    
    def get_audio_csv_file_pairs(self) -> List[Tuple[str, str]]:
        """
        Retrieves all the .wav and .csv file pairs that have the same base name from the folder.

        Returns:
        - A list of tuples, each containing a pair of .wav and .csv file paths.
        """
        i=0
        if not (self.phone_bool == False and self.word_bool == False and self.sentence_bool == False):
            for path in self.folder_path:
                self.label_path = os.path.basename(path)
                wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

                # Find pairs based on the base name (without the extension)
                file_pairs = []
                for wav_file in wav_files:
                    base_name = os.path.splitext(wav_file)[0]
                    corresponding_csv = base_name + '.csv'
                    if corresponding_csv in csv_files:
                        i+=1
                        self.process_csv(os.path.join(path, wav_file),os.path.join(path, corresponding_csv))
    
            return file_pairs

    def find_real_start_end(self,signal, sample_rate, window_size_ms=20, threshold=0.01, label=None):
        """
        Adjust the start and end of the word based on the rolling standard deviation.
        
        Parameters:
        - signal: The audio signal (1D numpy array).
        - sample_rate: The sample rate of the audio signal.
        - window_size_ms: The size of the window for computing the rolling standard deviation (default is 20ms).
        - threshold: The threshold for detecting significant changes in the standard deviation.
        
        Returns:
        - adjusted_start: The adjusted start time (in samples).
        - adjusted_end: The adjusted end time (in samples).
        """
        global sum_length
        # Convert window size from milliseconds to samples
        window_size = int((window_size_ms / 1000) * sample_rate)
        
        # Compute the rolling standard deviation
        rolling_std_dev = rolling_std(signal, window_size)

        """Ploting"""
        # # Plot the original signal and the rolling standard deviation
        # plt.figure(figsize=(14, 6))
        # plt.plot(signal, label='Original Signal', alpha=0.75)
        # plt.plot(np.arange(window_size, window_size + len(rolling_std_dev)), rolling_std_dev, label='Rolling Std Dev', color='orange')
        # plt.title('Original Signal and Rolling Standard Deviation of '+label)
        # plt.xlabel('Sample Index')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.grid()
        # plt.show()
        # # Plot the rolling standard deviation and threshold
        # plt.plot(rolling_std_dev, label='Rolling Std Dev')
        # plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold (0.01)')
        # plt.title('Rolling Standard Deviation and Threshold')
        # plt.xlabel('Frame')
        # plt.ylabel('Standard Deviation')
        # plt.legend()
        # plt.show()

        
        # Find the real start: from the beginning to where the std dev exceeds the threshold
        for i in range(len(rolling_std_dev)):
            if rolling_std_dev[i] > threshold:
                adjusted_start = i + window_size // 2  # Move to the middle of the window
                break
        else:
            adjusted_start = 0  # Fallback to the very start if no threshold exceeded
        
        # Find the real end: from the end to where the std dev exceeds the threshold (reverse search)
        for i in range(len(rolling_std_dev) - 1, -1, -1):
            if rolling_std_dev[i] > threshold:
                adjusted_end =  i + window_size // 2  # Move to the middle of the window
                break
        else:
            adjusted_end = len(signal)  # Fallback to the very end if no threshold exceeded
        
        if ((adjusted_end-adjusted_start)>self.maximum_word_length):
            self.maximum_word_length = (adjusted_end-adjusted_start)
            print("Maximum word length: ",self.maximum_word_length)
        sum_length += (adjusted_end-adjusted_start)
        return adjusted_start, adjusted_end

    def process_csv(self,wav_file,csv_file):
        global word_lengths_by_webmouse

        # Load the audio using librosa
        audio_data, sample_rate = librosa.load(wav_file, sr=None)
        # Normalize the audio to the range [-1, 1]
        max_amplitude = np.max(np.abs(audio_data))  # Find the maximum absolute amplitude
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude  # Normalize the audio by its maximum value

        # Load the CSV and process the word segments
        with open(csv_file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            word_segment = []
            word_start = None
            word_label = None
            dividing_word = False
            beginning = True
            current_word = None
            last_word = None
            word_fin = False

            # Make a copy of the sentences so we can keep track of unprocessed sentences
            next(reader)
            for row in reader:
                current_word = row[5]
                if beginning:
                    if current_word == '':
                        continue
                    else:
                        beginning = False
                        last_word = current_word
                if word_fin:
                    last_word = current_word
                    word_fin = False      
                
                duration = float(row[1])
                if(self.get_buffer):
                    start_time = (float(row[0]) - (self.buffer*sample_rate))
                    end_time = (start_time + duration + (self.buffer*sample_rate))
                else:
                    start_time = float(row[0])
                    end_time = start_time + duration

                #check if the phone is RoI
                if row[3] in self.phones and self.phone_bool:
                    if(start_time <0):
                        start_time = 0
                    segment = audio_data[int(start_time):int(end_time)]
                    if(self.downsample):
                        segment = librosa.resample(segment, orig_sr=self.org_sample_rate, target_sr=self.target_sr)
                        
                    self.phone_segments.append(
                        AudioSegment(start_time=start_time, 
                                    end_time=end_time, 
                                    audio_data=segment, 
                                    sample_rate= row[5], #here we save the word the phone comes from 
                                    label=row[3],# herre we save what phone type it is
                                    label_path=self.label_path,#sigmatis or normal
                                    path = wav_file)#path to wav file
                    )
                    
                # Pause handling: Pauses should be included in sentences, not treated as a break
                if current_word != last_word or current_word == '':
                    if word_segment:
                        # If we're in a sentence after "dividing_word"                      
                        if not dividing_word:
                            end_time = start_time  # because the word ends at the beginning of the pause but 5 ms already substracted and then the 5ms on top                
                            if self.word_bool:
                                ###make rolling std
                                word_start = (int(word_start-(self.buffer_word*sample_rate)))#add the windowing to search for end and beginning
                                end_time = (int(end_time+(self.buffer_word*sample_rate)))
                                if(word_start <0):
                                    word_start = 0
                                segment = audio_data[int(word_start):int(end_time)]
                                adjusted_start, adjusted_end = self.find_real_start_end(segment, sample_rate,label=word_label)
                                adjusted_segment = audio_data[(int(word_start+adjusted_start)):(int(word_start+adjusted_end))]
                                ###till here
                                if(self.downsample):
                                    adjusted_segment = librosa.resample(adjusted_segment, orig_sr=self.org_sample_rate, target_sr=self.target_sr)
                                if(adjusted_segment.size >= 2048):
                                    #print(adjusted_segment.size)
                                    self.word_segments.append(
                                        AudioSegment(start_time=int(word_start+adjusted_start), 
                                                    end_time=int(word_start+adjusted_end), 
                                                    audio_data=adjusted_segment, 
                                                    sample_rate=self.target_sr, 
                                                    label=word_label,
                                                    label_path=self.label_path,
                                                    path = wav_file)
                                    )
                            # Check if we are past the word "Xylophone"
                            if word_label == self.dividing_word:
                                dividing_word = True
                            word_segment = []
                            word_fin = True
                            word_start = None
                            word_label = None
                else:#it is no pause but a phone (word part)
                    # We are in the middle of a word or at the beginning
                    if word_start is None: #at the beginning of a word
                        word_start = start_time
                        last_word = current_word
                    word_label = row[5]  # Extract word label from the CSV
                    word_segment.append((start_time, end_time))

            self.audio_data = None
            print(f"Audio {wav_file} processed with {np.shape(self.phone_segments)} phones, {np.shape(self.word_segments)} words and {np.shape(self.sentence_segments)} sentences.")
    

    def save_segments_to_pickle(self,audio_segments: List[AudioSegment], filename: str):
        """
        Save the list of AudioSegment objects into a Pickle file.
        
        Parameters:
        - audio_segments: List of AudioSegment objects to save.
        - filename: The name of the Pickle file to save the data.
        """
        with open(filename, 'wb') as file:
            # Use pickle to dump the data into the file
            pickle.dump(audio_segments, file)

        print(f"Data saved to {filename}.")
    
    def load_segments_from_pickle(self,filename: str) -> List[AudioSegment]:
        """
        Load the AudioSegment objects from a Pickle file.
        
        Parameters:
        - filename: The name of the Pickle file to load data from.
        
        Returns:
        - A list of AudioSegment objects.
        """
        with open(filename, 'rb') as file:
            # Use pickle to load the data from the file
            audio_segments = pickle.load(file)

        print(f"Data loaded from {filename}.")
        return audio_segments
    
    def create_dataclass_words(self) -> List[AudioSegment]:
        return self.word_segments
    
    def create_dataclass_sentences(self) -> List[AudioSegment]:
       
        return self.sentence_segments
    
    def create_dataclass_phones(self) -> List[AudioSegment]:
        
        return self.phone_segments

def get_box_length(words_segments):
    label_count = {}
    global sum_length
    for word_segment in words_segments:
        label = word_segment.label_path
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    # Display the count for each label
    print(label_count )
    # Collect the word lengths per file (group by file paths)
    word_lengths_by_file = {}

    # Calculate word lengths for each word and group them by file path
    for word_segment in words_segments:
        sum_length += (word_segment.end_time - word_segment.start_time)
        word_length = (word_segment.end_time - word_segment.start_time) / 44100 #hier durch das teilen weil ich die marken abspeichere von dem audio das auf 44100 gesampled wurde.
        # if word_length > 1.2:
        #     # Plot the original signal and the rolling standard deviation
        #     leng = str(int(word_segment.end_time - word_segment.start_time))
        #     plt.figure(figsize=(14, 6))
        #     plt.plot(word_segment.audio_data, label='Original Signal', alpha=0.75)
        #     plt.title(str(word_length) + "    " + word_segment.label + "  "+ word_segment.path + leng)
        #     plt.xlabel('Time')
        #     plt.ylabel('Amplitude')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()
            
        if word_segment.label_path not in word_lengths_by_file:
            word_lengths_by_file[word_segment.label_path] = []
        word_lengths_by_file[word_segment.label_path].append(word_length)

    # Create a boxplot for word lengths by file
    plt.figure(figsize=(12, 6))
    plt.boxplot([word_lengths for word_lengths in word_lengths_by_file.values()], labels=word_lengths_by_file.keys(),vert=False)
    plt.title("Distribution of Word Lengths by Label")
    plt.ylabel("Label")
    plt.xlabel("Word Length (seconds)")
    plt.tight_layout()

    # Show the boxplot
    plt.show()

def find_pairs(audio_segments,phones_segments):
    """
    Takes a word which can be choosen by indices and searches for the correspüonding word in sig or normal. 
    Can also find all corresponding phones for a word.
    """
    sigmatism = None
    normal = None
    phones =["z","s","Z","S","ts"]
    phones_list_normal = []
    phones_list_sigmatism = []
    segment = audio_segments[2]###choose word here
    
    if segment.label_path == "sigmatism":
        print("It is Sigmatism")
        sigmatism = segment
        #get path from other file with normal speech
        matching_path = segment.path.replace("sigmatism", "normal")
        base, ext = os.path.splitext(matching_path)
        path = f"{base[:-4]}{ext}"
        print("PATH:",path)
        for normal in audio_segments:
            if (normal.label_path == "normal" and
                normal.label == segment.label and
                normal.path == path):
                print("Found normal pair")

                if(phones_segments):
                    for phone in phones_segments:
                        if (phone.label_path == "normal" and
                            phone.label in phones and
                            phone.path == path and
                            phone.sample_rate == sigmatism.label):
                            phones_list_normal.append(phone)

                        if (phone.label_path == "sigmatism" and
                            phone.label in phones and
                            phone.path == segment.path and
                            phone.sample_rate == sigmatism.label):
                            phones_list_sigmatism.append(phone)
                    return sigmatism, normal, phones_list_normal, phones_list_sigmatism
                return sigmatism, normal, phones_list_normal, phones_list_sigmatism

    
    if segment.label_path == "normal":
        print("It is Normal")
        normal =segment
        matching_path = segment.path.replace("normal", "sigmatism")
        base, ext = os.path.splitext(matching_path)
        path = f"{base}_sig{ext}"
        for sigmatism in audio_segments:
            if (sigmatism.label_path == "sigmatism" and
                sigmatism.label == normal.label and
                sigmatism.path == path):
                print("Found sigmatism pair")
                if(phones_segments):
                    for normal_phone in phones_segments:
                        if (normal_phone.label_path == "normal" and
                            normal_phone.label in phones and
                            normal_phone.path == normal.path and
                            normal_phone.sample_rate == sigmatism.label):
                            phones_list_normal.append(normal_phone)

                        if (normal_phone.label_path == "sigmatism" and
                            normal_phone.label in phones and
                            normal_phone.path == path and
                            normal_phone.sample_rate == sigmatism.label):
                            phones_list_sigmatism.append(normal_phone)
                    return sigmatism, normal, phones_list_normal, phones_list_sigmatism 
                return sigmatism, normal, phones_list_normal, phones_list_sigmatism 


    # If no pair is found, return None
    print("ERROR...............................................ERROR")
    return None, None,None,None

if __name__ == "__main__":
    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True, downsample=True)
   
    #phones_segments = loader.create_dataclass_phones()
    #words_segments = loader.create_dataclass_words()
    # sentences_segments = loader.create_dataclass_sentences()
    #loader.save_segments_to_pickle(phones_segments, "phones_atleast2048long_24kHz.pkl")
    #loader.save_segments_to_pickle(words_segments, "words_atleast2048.pkl")
    # loader.save_segments_to_pickle(sentences_segments, "sentences_segments.pkl")
    phones_segments = loader.load_segments_from_pickle("phones_atleast2048long_24kHz.pkl")
    words_segments = loader.load_segments_from_pickle("words_atleast2048long_24kHz.pkl")
    print(words_segments[0])
    #filtered_words = filter_and_pickle_audio_segments(words_segments)
    # sentences_segments = loader.load_segments_from_pickle("sentences_segments.pkl")
    # biggest_sample=0
    # # Calculate word lengths for each word and group them by fil path
    # for word_segment in words_segments:
    #     if(biggest_sample<word_segment.audio_data.shape[1]):
    #         print(word_segment.audio_data.shape)
    #         biggest_sample = word_segment.audio_data.shape[1]
    #         print(biggest_sample,word_segment.label)
    # print("biggest sample: ",biggest_sample)
    # sum_length =0
    
    #get_box_length(words_segments)
    # max_length = max([words.audio_data.shape[1] for words in words_segments]) #maximum of all mfccs
    # print("max_length: ",max_length)
    # print("shape",np.shape(words_segments))
    sigmatism, normal, phones_list_normal, phones_list_sigmatism = find_pairs(words_segments,phones_segments)
    #print(np.shape(phones_list_normal),np.shape(phones_list_sigmatism),sigmatism.label)
    #plotting.plot_mel_spectrogram(normal)
    #plotting.plot_mfcc_and_mel_spectrogram(sigmatism)
    plotting.plot_mel_spectrogram(normal,phones_list_normal)
    #plotting.compare_spectral_envelopes(sigmatism, normal)

   
   

  