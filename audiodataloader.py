#train GMM here to use it in paperimplementation
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

sum_length =0

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  
    label_path: str
    path: str

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
        self.target_sr = 32000
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
            sentence_start = None
            sentence_end = None
            sentence_label = None
            current_sentence = None
            dividing_word = False
            sentence_will_end = False
            beginning = True
            current_word = None
            last_word = None
            word_fin = False

            # Make a copy of the sentences so we can keep track of unprocessed sentences
            remaining_sentences = self.sentences.copy()
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
                                    sample_rate= sample_rate, 
                                    label=row[3],
                                    label_path=self.label_path,
                                    path = wav_file)
                    )
                    
                # Pause handling: Pauses should be included in sentences, not treated as a break
                if current_word != last_word or current_word == '':
                    if word_segment:
                        # If we're in a sentence after "dividing_word"
                        if not dividing_word and not current_sentence:                       
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
                                    self.word_segments.append(
                                        AudioSegment(start_time=int(word_start+adjusted_start), 
                                                    end_time=int(word_start+adjusted_end), 
                                                    audio_data=adjusted_segment, 
                                                    sample_rate=sample_rate, 
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
                        elif sentence_will_end:
                            sentence_end = start_time # because the word ends at the beginning of the pause
                            # Append the sentence segment and reset
                            if self.sentence_bool:
                                segment = audio_data[int(sentence_start):int(sentence_end)]
                                if(self.downsample):
                                        segment = librosa.resample(segment, orig_sr=self.org_sample_rate, target_sr=self.target_sr)
                                self.sentence_segments.append(
                                    AudioSegment(start_time=sentence_start, 
                                                end_time=sentence_end, 
                                                audio_data=segment, 
                                                sample_rate=sample_rate, 
                                                label=sentence_label,
                                                label_path=self.label_path,
                                                path = wav_file)
                                )
                            remaining_sentences.remove(current_sentence)  # Remove the processed sentence
                            current_sentence = None  
                            sentence_will_end = False
                else:#it is no pause but a phone (word part)
                    # We are in the middle of a word or at the beginning
                    if word_start is None: #at the beginning of a word
                        word_start = start_time
                        last_word = current_word
                    word_label = row[5]  # Extract word label from the CSV
                    word_segment.append((start_time, end_time))

                    # Check if we are in the sentence section (after Xylophone)
                    if dividing_word:
                        if current_sentence is None:
                            # Now search through remaining sentences instead of starting from scratch
                            for sentence in remaining_sentences:
                                if word_label == sentence[0]:  # Match the first word of a sentence
                                    current_sentence = sentence
                                    sentence_start = start_time
                                    sentence_label = sentence[0] + " " + sentence[1]  
                                    break
                        if current_sentence and word_label == current_sentence[1]:  # Last word of sentence
                            sentence_will_end = True


            # Append any remaining word or sentence if any
            if word_segment:
                if dividing_word and current_sentence:
                    if self.sentence_bool:
                        segment = audio_data[int(sentence_start):int(sentence_end)]
                        if(self.downsample):
                            segment = librosa.resample(segment, orig_sr=self.org_sample_rate, target_sr=self.target_sr)
                        self.sentence_segments.append(
                            AudioSegment(start_time=sentence_start, 
                                        end_time=sentence_end, 
                                        audio_data=segment, 
                                        sample_rate=sample_rate, 
                                        label=sentence_label,
                                        label_path=self.label_path,
                                        path = wav_file)
                        )
                elif not dividing_word:
                    if self.word_bool:
                        ###make rolling std
                        word_start = (int(word_start-(self.buffer_word*sample_rate)))
                        end_time = (int(end_time+(self.buffer_word*sample_rate)))
                        segment = audio_data[int(word_start):int(end_time)]
                        adjusted_start, adjusted_end = self.find_real_start_end(segment, sample_rate,word_label)
                        adjusted_segment = audio_data[(int(word_start+adjusted_start)):(int(word_start+adjusted_end))]
                        ###till here
                        if(self.downsample):
                            adjusted_segment = librosa.resample(adjusted_segment, orig_sr=self.org_sample_rate, target_sr=self.target_sr)
                        self.word_segments.append(
                            AudioSegment(start_time=int(word_start+adjusted_start), 
                                        end_time=int(word_start+adjusted_end), 
                                        audio_data=adjusted_segment, 
                                        sample_rate=sample_rate, 
                                        label=word_label,
                                        label_path=self.label_path,
                                        path = wav_file)
                        )
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
    print(label_count)
    # Collect the word lengths per file (group by file paths)
    word_lengths_by_file = {}

    # Calculate word lengths for each word and group them by file path
    for word_segment in words_segments:
        sum_length += (word_segment.end_time - word_segment.start_time)
        word_length = (word_segment.end_time - word_segment.start_time) / word_segment.sample_rate
        if word_length > 1.2:
            # Plot the original signal and the rolling standard deviation
            leng = str(int(word_segment.end_time - word_segment.start_time))
            plt.figure(figsize=(14, 6))
            plt.plot(word_segment.audio_data, label='Original Signal', alpha=0.75)
            plt.title(str(word_length) + "    " + word_segment.label + "  "+ word_segment.path + leng)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()
            plt.show()
            
        if word_segment.label_path not in word_lengths_by_file:
            word_lengths_by_file[word_segment.label_path] = []
        word_lengths_by_file[word_segment.label_path].append(word_length)

    # Create a boxplot for word lengths by file
    plt.figure(figsize=(12, 6))
    plt.boxplot([word_lengths for word_lengths in word_lengths_by_file.values()], labels=word_lengths_by_file.keys())
    plt.title("Distribution of Word Lengths by File")
    plt.xlabel("Files")
    plt.ylabel("Word Length (seconds)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Show the boxplot
    plt.show()



if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True, downsample=True)
    # # Sample signal data
    # np.random.seed(0)
    # signal = np.random.randn(100000)  # Large array for performance testing
    # window_size = 100
    
    # # Timing the Numba implementation
    # numba_time = timeit.timeit(lambda: rolling_std_numba(signal, window_size), number=10)
    # print(f"Numba implementation time: {numba_time / 10:.5f} seconds")
    # # Timing the Pandas implementation
    # pandas_time = timeit.timeit(lambda: loader.rolling_std(signal, window_size), number=10)
    # print(f"Pandas implementation time: {pandas_time / 10:.5f} seconds")


    #phones_segments = loader.create_dataclass_phones()
    words_segments = loader.create_dataclass_words()
    # sentences_segments = loader.create_dataclass_sentences()
    # loader.save_segments_to_pickle(phones_segments, "phones_segments.pkl")
    loader.save_segments_to_pickle(words_segments, "all_words_downsampled_to_32kHz.pkl")
    # loader.save_segments_to_pickle(sentences_segments, "sentences_segments.pkl")
    # phones_segments = loader.load_segments_from_pickle("phones_segments.pkl")
    # words_segments = loader.load_segments_from_pickle("all_words_downsampled_to_8kHz.pkl")
    #filtered_words = filter_and_pickle_audio_segments(words_segments)
    # sentences_segments = loader.load_segments_from_pickle("sentences_segments.pkl")
    #print(np.shape(phones_segments))
    biggest_sample=0
    # Calculate word lengths for each word and group them by file path
    for word_segment in words_segments:
        if(biggest_sample<word_segment.audio_data.size):
            biggest_sample = word_segment.audio_data.size
    print("biggest sample: ",biggest_sample)
    print("Avg Length 1: ",sum_length/np.shape(words_segments)[0])
    sum_length =0
    #get_box_length(words_segments)
    print("Avg Length: ",sum_length/np.shape(words_segments)[0])
    print(np.shape(words_segments))
    #print(np.shape(sentences_segments),type(sentences_segments))
    
    print("WORDS:::::::::::::::::::::::::::::::")
    for i in range(5):
        print((words_segments[i].end_time-words_segments[i].start_time)/words_segments[i].sample_rate,words_segments[i].label_path)
        
    # print("PHONES::::::::::::::::::::::::::::::")
    # for i in range(5):
    #     print((phones_segments[i].end_time-phones_segments[i].start_time)/phones_segments[i].sample_rate,phones_segments[i].label_path)
   
    # print("SENTENCES:::::::::::::::::::::::::::")
    # for i in range(5):
    #     print((sentences_segments[i].end_time-sentences_segments[i].start_time)/sentences_segments[i].sample_rate,sentences_segments[i].label_path)

    
