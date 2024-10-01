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

        

"""
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import json
import os

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  
    label_path: str

class AudioDataLoader:
    def __init__(self, config_file: str, phone_data: bool = False, word_data: bool = False, sentence_data: bool = False):
        self.phone_bool = phone_data
        self.word_bool = word_data
        self.sentence_bool = sentence_data
        self.load_config(config_file)
        self.word_segments = []
        self.sentence_segments = []
        self.phone_segments = []
        self.phones =["z","s","Z","S","ts"]
        self.files = self.get_audio_csv_file_pairs()
        self.folder_path = None
        self.dividing_word
        self.label_path = None


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

    def process_csv(self,wav_file,csv_file):
        # Load the audio using librosa
        audio_data, sample_rate = librosa.load(wav_file, sr=None)
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
                if row[3] in self.phones and self.phone_bool:
                    segment = audio_data[int(start_time):int(end_time)]
                    self.phone_segments.append(
                        AudioSegment(start_time=start_time, 
                                    end_time=end_time, 
                                    audio_data=segment, 
                                    sample_rate= sample_rate, 
                                    label=row[3],
                                    label_path=self.label_path)
                    )
                    
                # Pause handling: Pauses should be included in sentences, not treated as a break
                if token == '-1':
                    if word_segment:
                        # If we're in a sentence after "dividing_word"
                        if not dividing_word and not current_sentence:                       
                            if not dividing_word:
                                end_time = start_time  # because the word ends at the beginning of the pause
                                if self.word_bool:
                                    segment = audio_data[(int(word_start)):(int(end_time))]
                                    self.word_segments.append(
                                        AudioSegment(start_time=word_start, 
                                                    end_time=end_time, 
                                                    audio_data=segment, 
                                                    sample_rate=sample_rate, 
                                                    label=word_label,
                                                    label_path=self.label_path)
                                    )
                                # Check if we are past the word "Xylophone"
                                if word_label == self.dividing_word:
                                    dividing_word = True
                                word_segment = []
                                word_start = None
                                word_label = None
                        elif sentence_will_end:
                            sentence_end = start_time # because the word ends at the beginning of the pause
                            # Append the sentence segment and reset
                            if self.sentence_bool:
                                segment = audio_data[int(sentence_start):int(sentence_end)]
                                self.sentence_segments.append(
                                    AudioSegment(start_time=sentence_start, 
                                                end_time=sentence_end, 
                                                audio_data=segment, 
                                                sample_rate=sample_rate, 
                                                label=sentence_label,
                                                label_path=self.label_path)
                                )
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
                        self.sentence_segments.append(
                            AudioSegment(start_time=sentence_start, 
                                        end_time=sentence_end, 
                                        audio_data=segment, 
                                        sample_rate=sample_rate, 
                                        label=sentence_label,
                                        label_path=self.label_path)
                        )
                elif not dividing_word:
                    if self.word_bool:
                        segment = audio_data[int(word_start):int(end_time)]
                        self.word_segments.append(
                            AudioSegment(start_time=word_start, 
                                        end_time=end_time, 
                                        audio_data=segment, 
                                        sample_rate=sample_rate, 
                                        label=word_label,
                                        label_path=self.label_path)
                        )
            self.audio_data = None
            print(f"Audio {wav_file} processed with {np.shape(self.phone_segments)} phones, {np.shape(self.word_segments)} words and {np.shape(self.sentence_segments)} sentences.")
            

    def create_dataclass_words(self) -> List[AudioSegment]:
        return self.word_segments
    
    def create_dataclass_sentences(self) -> List[AudioSegment]:
       
        return self.sentence_segments
    
    def create_dataclass_phones(self) -> List[AudioSegment]:
        
        return self.phone_segments

if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= True, phone_data= False, sentence_data= True)

    phones_segments = loader.create_dataclass_phones()
    print(np.shape(phones_segments))
    words_segments = loader.create_dataclass_words()
    print(np.shape(words_segments))
    sentences_segments = loader.create_dataclass_sentences()
    print(np.shape(sentences_segments),type(sentences_segments))
    print("PHONES::::::::::::::::::::::::::::::")
    for i in phones_segments:
        print((i.end_time-i.start_time)/i.sample_rate,i.label_path)
    
    print("WORDS:::::::::::::::::::::::::::::::")

    for i in words_segments:
        print((i.end_time-i.start_time)/i.sample_rate,i.label_path)

    print("SENTENCES:::::::::::::::::::::::::::")

    for i in sentences_segments:
        print((i.end_time-i.start_time)/i.sample_rate,i.label_path)
