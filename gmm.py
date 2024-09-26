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
Schauen ob es ein vorimplementieres model gibt.
1. Vor und nach der Pause noch bisl mit dazunehmen. 
2. Die Sätze hardcoden weil man weis mit welchem wort die anfangen und aufhören. -> Chatgpt

"""
import csv
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union

@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    label: str  # Word or Sentence

class AudioDataLoader:
    def __init__(self, csv_file: str, audio_file: str):
        self.csv_file = csv_file
        self.audio_file = audio_file
        self.word_segments = []
        self.sentence_segments = []
        self.sample_rate = None
        self.audio_data = None

    def load_audio(self):
        # Load the audio using librosa
        self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
        print(f"Audio loaded with sample rate {self.sample_rate}.")

    def process_csv(self):
        # Load the CSV and process the word segments
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            word_segment = []
            word_start = None
            word_label = None
            after_xylophone = False
            beginning = True

            for row in reader:
                #because first label of file is a pause.
                if(beginning):
                    beginning = False
                    continue
                start_time = float(row[0])
                duration = float(row[1])
                token = row[2] 
                end_time = start_time + duration

                # If the token is a pause (-1), this means end of a word or a pause
                if token == '-1':
                    if word_segment:
                        # Check if we're in the sentence segment (after "Xylophone")
                        if after_xylophone:
                            self.sentence_segments.append((word_start, end_time, word_label))
                        else:
                            end_time = start_time#because the words ends at the beginning of the pause
                            self.word_segments.append((word_start, end_time, word_label))
                        word_segment = []
                        word_start = None
                        word_label = None
                else:
                    # We are in the middle of a word or beginning
                    if word_start is None:
                        word_start = start_time 
                    word_label = row[5]   
                    word_segment.append((start_time, end_time))

                # Check if we are past the word "Xylophone"
                if not after_xylophone and word_label == "Xylophone":
                    after_xylophone = True

            # Append the last word or sentence if any
            if word_segment:
                if after_xylophone:
                    self.sentence_segments.append((word_start, end_time, word_label))
                else:
                    self.word_segments.append((word_start, end_time, word_label))

    def get_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        start_sample = int(start_time)
        end_sample = int(end_time)
        return self.audio_data[start_sample:end_sample]

    def create_dataclass_entries(self) -> List[AudioSegment]:
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

if __name__ == "__main__":

    loader = AudioDataLoader(csv_file='Tonaufnahmen\speaker1gtnurwords.csv', audio_file='Tonaufnahmen\speaker1gtnurwords.wav')
    loader.load_audio()
    loader.process_csv()
    audio_segments = loader.create_dataclass_entries()
   
    print(audio_segments[0].audio_data )
