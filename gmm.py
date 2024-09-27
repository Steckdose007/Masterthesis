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

"""
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import json

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  # Word or Sentence

class AudioDataLoader:
    def __init__(self, csv_file: str, audio_file: str, config_file: str):
        self.csv_file = csv_file
        self.audio_file = audio_file
        self.word_segments = []
        self.sentence_segments = []
        self.phone_segments = []
        self.phones =["z","s","Z","S","ts"]
        self.sample_rate = None
        self.audio_data = None
        self.load_config(config_file)
        self.load_audio()
        self.process_csv()


    def load_audio(self):
        # Load the audio using librosa
        self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
        print(f"Audio loaded with sample rate {self.sample_rate}.")

    def load_config(self, config_file: str):
            """Load sentences and dividing word from the JSON config file."""
            with open(config_file, 'r') as file:
                config = json.load(file)
            
            self.sentences = config["sentences"]  # Load the sentences
            self.dividing_word = config["dividing_word"]  # Load the dividing word (e.g., "Xylophone")

    def process_csv(self):
        # Load the CSV and process the word segments
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            word_segment = []
            word_start = None
            word_label = None
            sentence_start = None
            sentence_end = None
            sentence_label = None
            current_sentence = None
            after_xylophone = False
            sentence_will_end = False
            beginning = True

            # Make a copy of the sentences so we can keep track of unprocessed sentences
            remaining_sentences = self.sentences.copy()

            for row in reader:
                if beginning:
                    beginning = False
                    continue

                start_time = float(row[0])
                duration = float(row[1])
                token = row[2] 
                end_time = start_time + duration

                #check if the phone is RoI
                if row[3] in self.phones:
                    self.phone_segments.append((start_time, end_time, row[3]))
                    
                # Pause handling: Pauses should be included in sentences, not treated as a break
                if token == '-1':
                    if word_segment:
                        # If we're in a sentence after "Xylophone"
                        if not after_xylophone and not current_sentence:                       
                            if not after_xylophone:
                                end_time = start_time  # because the word ends at the beginning of the pause
                                self.word_segments.append((word_start, end_time, word_label))
                                # Check if we are past the word "Xylophone"
                                if word_label == "Xylophon":
                                    after_xylophone = True
                                word_segment = []
                                word_start = None
                                word_label = None
                        elif sentence_will_end:
                            sentence_end = start_time # because the word ends at the beginning of the pause
                            # Append the sentence segment and reset
                            self.sentence_segments.append((sentence_start, sentence_end, sentence_label))
                            remaining_sentences.remove(current_sentence)  # Remove the processed sentence
                            current_sentence = None  
                            sentence_will_end = False
                else:#it is no pause but a phone (word part)
                    # We are in the middle of a word or at the beginning
                    if word_start is None: #at the beginning of a word
                        word_start = start_time
                    word_label = row[5]  # Extract word label from the CSV
                    word_segment.append((start_time, end_time))

                    # Check if we are in the sentence section (after Xylophone)
                    if after_xylophone:
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
                if after_xylophone and current_sentence:
                    self.sentence_segments.append((sentence_start, sentence_end, sentence_label))
                else:
                    self.word_segments.append((word_start, end_time, word_label))
                    
            print(f"Audio processed with {np.shape(self.phone_segments)} phones, {np.shape(self.word_segments)} words and {np.shape(self.sentence_segments)} sentences.")
            print(f"Audio ready to rumbel.")


    def get_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        start_sample = int(start_time)
        end_sample = int(end_time)
        return self.audio_data[start_sample:end_sample]

    def create_dataclass_words(self) -> List[AudioSegment]:
        data_entries = []
        
        # Process word segments
        for start_time, end_time, label in self.word_segments:
            segment = self.get_audio_segment(start_time, end_time)
            data_entries.append(
                AudioSegment(start_time=start_time, 
                             end_time=end_time, 
                             audio_data=segment, 
                             sample_rate=self.sample_rate, 
                             label=label)
            )
        
        return data_entries
    
    def create_dataclass_sentences(self) -> List[AudioSegment]:
        data_entries = []
        
        # Process sentence segments
        for start_time, end_time, label in self.sentence_segments:
            segment = self.get_audio_segment(start_time, end_time)
            data_entries.append(
                AudioSegment(start_time=start_time, 
                             end_time=end_time, 
                             audio_data=segment, 
                             sample_rate=self.sample_rate, 
                             label=label)
            )
        
        return data_entries
    
    def create_dataclass_phones(self) -> List[AudioSegment]:
        data_entries = []
        
        # Process phone segments
        for start_time, end_time, label in self.phone_segments:
            segment = self.get_audio_segment(start_time, end_time)
            data_entries.append(
                AudioSegment(start_time=start_time, 
                             end_time=end_time, 
                             audio_data=segment, 
                             sample_rate=self.sample_rate, 
                             label=label)
            )
        
        return data_entries

if __name__ == "__main__":

    loader = AudioDataLoader(csv_file='Tonaufnahmen\speaker4gt.csv', audio_file='Tonaufnahmen\speaker4gt.wav', config_file='config.json')

    phones_segments = loader.create_dataclass_phones()
    print(np.shape(phones_segments))
    words_segments = loader.create_dataclass_words()
    print(np.shape(words_segments))
    sentences_segments = loader.create_dataclass_sentences()
    print(np.shape(sentences_segments))
