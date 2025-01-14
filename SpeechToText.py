import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audiodataloader import AudioDataLoader, AudioSegment
from torch.utils.data import DataLoader
import random
import soundfile as sf
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 


def interfere_whole_wav():
    target_sr = 16000
    # data1, sr = sf.read("Data/normal/Audio_1.wav")
    # data1 = librosa.resample(data1, orig_sr=sr, target_sr=target_sr)
    # print("Sample rate:", sr)
    # print("Data shape:", data1.shape)
    # print("Data type:", data1.dtype)

    # data, sr = sf.read("Data/common_voice_de_34922204.mp3")
    # data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    # print("Sample rate:", sr)
    # print("Data shape:", data.shape)
    # print("Data type:", data.dtype)


    # inputs = processor(data1, sampling_rate=16000, return_tensors="pt", padding=True)
    # with torch.no_grad():
    #     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    # predicted_ids = torch.argmax(logits, dim=-1)
    # predicted_sentences = processor.batch_decode(predicted_ids)

    # print("-" * 100)
    # print("Prediction:", predicted_sentences)

def interfere_segments(words_segments):
    dataset_length = len(words_segments)
    segment = words_segments[100]#random.randint(0, dataset_length - 1)]
    audio = segment.audio_data
    print(segment.label)
    print(segment.label_path)
    #audio = audio.astype(np.float32)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits  = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    print("output",logits.shape)
    print_outputs(logits,segment.label,segment.label_path)
    print("logits: ",type(logits))
    probs = F.softmax(logits, dim=-1)
    s_token_id = processor.tokenizer.convert_tokens_to_ids("S")
    s_probs = probs[0, :, s_token_id]

    print("Shape of s_probs:", s_probs.shape)
    # print("Probabilities of 'S' at each time step:", s_probs)


    predicted_ids = torch.argmax(logits , dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    print("-" * 100)
    print("Reference:", segment.label)
    print("Prediction:", predicted_sentences)

def print_outputs(output,label_word,label_path):
    logits = output[0]  # shape: [time_steps, vocab_size]
    """plot heatmap for all character"""
    # 1. Convert to numpy array for plotting
    logits_np = logits.detach().cpu().numpy()  # shape: (time_steps, vocab_size)
    # 2. Plot as a heatmap
    plt.figure(figsize=(12, 6))
    # We transpose so:
    #  - x-axis = time steps
    #  - y-axis = vocab indices
    # shape becomes (vocab_size, time_steps)
    plt.imshow(logits_np.T, aspect='auto', cmap='plasma', origin='lower')
    # Get vocabulary tokens
    vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))

    # Number of ticks (this will match vocab_size)
    num_vocab = len(vocab_tokens)

    # Set up the ticks on Y-axis at intervals (be careful with large vocab)
    plt.yticks(
        ticks=np.arange(num_vocab),
        labels=vocab_tokens,
        fontsize=6  # might need to reduce font size if it's a large vocab
    )
    plt.title(f"{label_word} Logits Heatmap with {label_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("Vocab Index")
    plt.colorbar(label="Logit Value")

    plt.tight_layout()
    plt.show()

    """Plot prediction for s over time"""
    # 1. Convert logits -> probabilities
    probs = F.softmax(logits, dim=-1)  # shape: [time_steps, vocab_size]

    # 2. Find token ID for "S"
    #    Check if "S" exists in the processor vocabulary (some models might use lowercase "s").
    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")
    if s_token_id is None:
        raise ValueError("Token 'S' not in vocabulary. Try 's' or check your model's vocab.")

    # 3. Extract probability of "S" across time
    s_probs = probs[:, s_token_id].cpu().numpy()  # shape: [time_steps]
    #s_probs_normalized = s_probs / np.sum(s_probs)

    # 4. Create a time axis (approximation)
    time_axis = np.arange(s_probs.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, s_probs, label="P('S')")
    plt.xlabel("Time Steps")
    plt.ylabel("Probability of 'S'")
    plt.title("Probability of 'S' Across Time Steps")
    plt.legend()
    plt.tight_layout()
    plt.show()

def area_under_curve(words_segments):
    # 1. Initialize dictionaries for AUC values
    auc_values = {"S": {"normal": [], "sigmatism": []},
                  "X": {"normal": [], "sigmatism": []},
                  "Z": {"normal": [], "sigmatism": []}}

    dataset_length = len(words_segments)

    # 2. Loop through words
    for i in tqdm(range(dataset_length), desc="Processing words"):
        segment = words_segments[i]
        audio = segment.audio_data
        if(audio.size <= 2048):
            continue
        label = segment.label_path  # "normal" or "sigmatism"
        word = segment.label.lower() # Word for filtering relevant characters (assumes lowercase)
        corrected_text = word.replace("ÃŸ", "s").replace("Ã¤", "ä").replace("Ã¼", "ü").replace("Ã¶", "ö")
        # 2.1 Preprocess audio
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

        # 2.2 Get logits and calculate probabilities
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        probs = F.softmax(logits[0], dim=-1)  # [time_steps, vocab_size]

        # 2.3 Extract probabilities and calculate AUC for characters present in the word
        for char in ["s", "x", "z"]:
            if char in corrected_text:  # Only process if the character is in the word
                token_id = processor.tokenizer.convert_tokens_to_ids(char)
                if token_id is None:
                    raise ValueError(f"Token '{char}' not found in vocabulary.")

                char_probs = probs[:, token_id].cpu().numpy()  # [time_steps]
                auc = np.sum(char_probs)  # Area under the curve (AUC)

                # 2.4 Store AUC in the corresponding dictionary
                if label == "normal":
                    auc_values[char.upper()]["normal"].append(auc)
                elif label == "sigmatism":
                    auc_values[char.upper()]["sigmatism"].append(auc)
                else:
                    raise ValueError(f"Unexpected label: {label}")

    # 3. Plot results
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]

        data = [sigmatism_aucs, normal_aucs]
        labels = ['Sigmatism', 'Normal']

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=labels, patch_artist=True, notch=True, showmeans=True)

        plt.title(f"Average Area Under Curve (AUC) for '{char}'")
        plt.ylabel("Normalized AUC")
        plt.tight_layout()
        plt.show()

def only_take_s(words_segments):
    # 2. Initialize lists for areas under curve
    probabilities_normal = []
    probabilities_sigmatism = []
    dataset_length = len(words_segments)
    # 3. Loop through words
    for i in tqdm(range(dataset_length), desc="Processing words"):          
        segment = words_segments[i]#random.randint(0, dataset_length - 1)]
        audio = segment.audio_data
        label = segment.label_path  # "normal" or "sigmatism"
        
        # 3.1 Preprocess audio
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        # 3.2 Get logits and calculate probabilities
        with torch.no_grad():
            outputs  = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        logits = outputs[0]  # [time_steps, vocab_size]
        probs = F.softmax(logits, dim=-1)  
        
        # 3.3 Extract probabilities for "S"
        s_token_id = processor.tokenizer.convert_tokens_to_ids("s")
        # Identify time steps where "S" is the most probable token
        predicted_ids = torch.argmax(probs, dim=-1)  # [time_steps]
        s_time_steps = torch.where(predicted_ids == s_token_id)[0]  # Indices where "S" is predicted

        # Collect probabilities for those time steps
        s_probs = probs[s_time_steps, s_token_id].cpu().numpy()  # Probabilities of "S" at those time steps
    
        # Collect probabilities for those time steps or save 0 if no "S" is predicted
        if len(s_time_steps) > 0:
            s_probs = probs[s_time_steps, s_token_id].cpu().numpy()  # Probabilities of "S" at those time steps
            avg_s_prob = np.mean(s_probs)  # Take the average if multiple "S"s are predicted
        else:
            avg_s_prob = 0.0  # No "S" predicted

        # Store in the appropriate list
        if label == "normal":
            probabilities_normal.append(avg_s_prob)
        elif label == "sigmatism":
            probabilities_sigmatism.append(avg_s_prob)
        else:
            raise ValueError(f"Unexpected label: {label}")

    # Prepare data for plotting
    data = [probabilities_normal, probabilities_sigmatism]
    labels = ["Normal", "Sigmatism"]
    print(data)
    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="lightblue", color="blue"), 
                medianprops=dict(color="red"))

    # Add labels and title
    plt.title("Probabilities of 'S' When Predicted as Most Likely")
    plt.ylabel("Probability of 'S'")
    plt.xlabel("Category")

    # Show the plot
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    loader = AudioDataLoader(config_file='config.json', word_data= False, phone_data= False, sentence_data= False, get_buffer=True)
    words_segments = loader.load_segments_from_pickle("words__16kHz.pkl")
    interfere_segments(words_segments)
    #area_under_curve(words_segments)
    #only_take_s(words_segments)


