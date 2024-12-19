import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audiodataloader import AudioDataLoader, AudioSegment
from torch.utils.data import DataLoader
import random
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
words_segments = loader.load_segments_from_pickle("words_atleast2048long_16kHz.pkl")
dataset_length = len(words_segments)

for i in range(100):
    segment = words_segments[random.randint(0, dataset_length - 1)]
    audio = segment.audio_data
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    print("-" * 100)
    print("Reference:", segment.label)
    print("Prediction:", predicted_sentences)