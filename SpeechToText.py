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
import csv
import pickle

def find_pairs(segment,phones_segments):

        phones =["z","s","Z","ts"]
        phones_list = []
       
        if(phones_segments):
            for phone in phones_segments:
                if (phone.label in phones and
                    phone.path == segment.path and
                    phone.sample_rate == segment.label):
                    phones_list.append(phone)
        return  phones_list


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
    """
    get prediction from STT for a word at index x
    the heatmap over time steps and the probability function of s over time steps are plottet. 
    """
    index = 5030
    dataset_length = len(words_segments)
    segment = words_segments[index]#random.randint(0, dataset_length - 1)]
    audio = segment.audio_data
    print("Word: ",segment.label)
    print("Label: ",segment.label_path)
    #audio = audio.astype(np.float32)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits  = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    print("output",logits.shape)
    print_outputs(logits,segment.label,segment.label_path)
    #print("logits: ",type(logits))
    probs = F.softmax(logits, dim=-1)
    print_outputs(probs,segment.label,segment.label_path)

    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")
    s_probs = probs[0, :, s_token_id]

    print("Shape of s_probs:", s_probs.shape)
    # print("Probabilities of 'S' at each time step:", s_probs)


    predicted_ids = torch.argmax(logits , dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    print("-" * 100)
    print("Reference:", segment.label)
    print("Prediction:", predicted_sentences)
    plt.show()

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

def area_under_curve(words_segments):
    """
    calculates the area under the curve of the prediction of the chars s,x,z over time steps for all words. 
    """
    # 1. Initialize dictionaries for AUC values
    auc_values = {"S": {"normal": [], "sigmatism": []},
                  "X": {"normal": [], "sigmatism": []},
                  "Z": {"normal": [], "sigmatism": []}}
    # To store per-word results
    per_word_auc = []
    dataset_length = len(words_segments)

    # 2. Loop through words
    #for i in tqdm(range(dataset_length), desc="Processing words"):
    for i in range(dataset_length):

        segment = words_segments[i]
        audio = segment.audio_data
        if(audio.size <= 2048):
            continue
        label = segment.label_path  # "normal" or "sigmatism"
        word = segment.label.lower() # Word for filtering relevant characters
        corrected_text = word.replace("ÃŸ", "s").replace("ãÿ", "s").replace("Ã¤", "ä").replace("ã¼", "ü").replace("Ã¼", "ü").replace("Ã¶", "ö").replace("ã¤", "ä").lower()
        # 2.1 Preprocess audio
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

        # 2.2 Get logits and calculate probabilities
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        probs = F.softmax(logits[0], dim=-1)  # [time_steps, vocab_size]

        # 2.3 Extract probabilities and calculate AUC for characters present in the word
        for char in ["s", "x", "z"]:
            if char in corrected_text:  # Only process if the character is in the word
                #num_s = corrected_text.count('char')
                token_id = processor.tokenizer.convert_tokens_to_ids(char)
                if token_id is None:
                    raise ValueError(f"Token '{char}' not found in vocabulary.")
                #print(token_id,char)
                char_probs = probs[:, token_id].cpu().numpy()  # [time_steps]
                auc = np.sum(char_probs)#/num_s  # Area under the curve (AUC)

                sum_logits = np.sum(logits[0][:, token_id].cpu().numpy())
                # 2.4 Store AUC in the corresponding dictionary
                if label == "normal":
                    auc_values[char.upper()]["normal"].append(auc)
                elif label == "sigmatism":
                    auc_values[char.upper()]["sigmatism"].append(auc)
                else:
                    raise ValueError(f"Unexpected label: {label}")
                # -- Store per-word AUC --
                per_word_auc.append({
                    "label": corrected_text,  #word for example "sonne"
                    "label_path": label, # sigmatism or normal
                    "path": segment.path, # which file it is from
                    "auc": float(auc),  
                    "heatmap": logits[0],
                    "logtis_AuC": sum_logits
                })

    #save
    with open("per_word_auc_values_deivided_by_nums.pkl", "wb") as f:
        pickle.dump(per_word_auc, f)


    # 3. Plot results
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]

        data = [sigmatism_aucs, normal_aucs]
        labels = ['Sigmatism', 'Normal']

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=labels, patch_artist=True, notch=True, showmeans=True)

        plt.title(f"Average Area Under Curve (AUC) for '{char}'")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        # Plot Normal data
        plt.hist(normal_aucs, bins=40, alpha=0.5, label='Normal')
        # Plot Sigmatism data
        plt.hist(sigmatism_aucs, bins=40, alpha=0.5, label='Sigmatism')
        
        plt.title('AUC Distributions: Normal vs. Sigmatism')
        plt.xlabel('AUC')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()

def area_under_curve_relative(words_segments, processor, model):
    """
    1) Get the full STT logits (heatmap) for each audio.
    2) Naively divide the time-frames among characters in the word.
    3) Only extract the portion for characters s, x, or z .
    4) Compute area under the curve (sum of probabilities) in that subrange.
    """

    # Dictionary for aggregated results
    auc_values = {
        "S": {"normal": [], "sigmatism": []},
        "X": {"normal": [], "sigmatism": []},
        "Z": {"normal": [], "sigmatism": []}
    }
    per_word_aucAttention2 = []

    # Characters of interest
    target_chars = ["s", "x", "z"]  

    #for segment in tqdm(words_segments, desc="Processing words"):
    for segment in words_segments:

        audio = segment.audio_data
        if audio.size <= 2048:
            continue

        label = segment.label_path  # "normal" or "sigmatism"
        word = segment.label.lower() # Word for filtering relevant characters (assumes lowercase)
        corrected_text = word.replace("ÃŸ", "ss").replace("ãÿ", "ss").replace("Ã¤", "ä").replace("ã¼", "ü").replace("Ã¼", "ü").replace("Ã¶", "ö").replace("ã¤", "ä").lower()
        print(word,corrected_text)
        # --- 1) Compute the full logits/probs (the "heatmap") ---ã¤ ãÿ ã¼
        inputs = processor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt", 
            padding=True
        )
        with torch.no_grad():
            logits = model(
                inputs.input_values, 
                attention_mask=inputs.attention_mask
            ).logits
        
        # logits shape: [time_steps, vocab_size]
        probs = F.softmax(logits[0], dim=-1)
        time_steps = probs.shape[0]

        fig = plt.figure(figsize=(12, 8))
        # Plot the audio waveform
        plt.imshow(logits[0].detach().cpu().numpy().T, aspect='auto', cmap='plasma', origin='lower')
        vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))
        num_vocab = len(vocab_tokens)
        plt.yticks(
            ticks=np.arange(num_vocab),
            labels=vocab_tokens,
            fontsize=6  # might need to reduce font size if it's a large vocab
        )
        plt.title(f"{corrected_text} Logits Heatmap with {label}")
        plt.xlabel("Time Steps")
        plt.ylabel("Vocab Index")
        plt.colorbar(label="Logit Value")

        """Attention2-----------------------------------------------------------------------------^
        """
        # --- 2) Naively map each character to a chunk of the frames ---
        num_chars = len(corrected_text)
        if num_chars == 0:
            continue

        # "Frames per character" in a uniform sense
        frames_per_char = time_steps / num_chars

        # For each occurrence of 's', 'x', or 'z' in the text
        # figure out which frames correspond to that character
        for idx, char in enumerate(corrected_text):
            # We only care about s/x/z
            if char not in target_chars:
                continue
            length = len(corrected_text)
            # SCH ans ST is not sigmatism
            if idx + 2 < length and corrected_text[idx+1:idx+3] == 'ch':
                continue
            elif idx + 1 < length and corrected_text[idx+1] == 't':
                continue
            # Start/end frames for this character
            start_frame = int(np.floor(idx * frames_per_char))
            end_frame = int(np.floor((idx + 1) * frames_per_char))

            # Safety clip to 0..time_steps
            start_frame = max(0, start_frame)
            end_frame = min(time_steps, end_frame)
            
            if start_frame >= end_frame:
                continue  # no frames to compute
            print("Start: ",start_frame)
            print("END:",end_frame)
            plt.axvline(x=start_frame, color='green', linestyle='--', label='Phone Start')
            plt.axvline(x=end_frame, color='red', linestyle='--', label='Phone End')

            # Find the token_id for the target char in the tokenizer
            token_id = processor.tokenizer.convert_tokens_to_ids(char)
            if token_id is None:
                # If not found in vocabulary, skip
                continue

            # Slice the probability subrange in the logits for this character
            sub_probs = probs[start_frame:end_frame, token_id].cpu().numpy()
            
            # --- 3) Compute AUC as the sum of probabilities in that range ---
            auc_char = float(np.sum(sub_probs))  # convert to float for cleanliness

            # --- 4) Store results in aggregated dictionary ---
            if label == "normal":
                auc_values[char.upper()]["normal"].append(auc_char)
            elif label == "sigmatism":
                auc_values[char.upper()]["sigmatism"].append(auc_char)
            else:
                # If you have other labels, handle them here
                pass

            # (Optional) store per-word data
            per_word_aucAttention2.append({
                "word": word,
                "label": label,
                "character": char,
                "position_in_word": idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "auc": auc_char,
                "sub_probs_all_vocab": probs[start_frame:end_frame, :].cpu().numpy()
            })
        plt.legend()
        plt.tight_layout()
        plt.show()
        """Attention 2 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        -----------------------------------------------------------------------------
        """

    # --- 5) Plot aggregated results ---
    # For each relevant char, show boxplot
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]

        if not (normal_aucs or sigmatism_aucs):
            continue  # skip if there's no data for this char

        plt.figure(figsize=(8, 6))
        plt.boxplot([sigmatism_aucs, normal_aucs],
                    labels=["Sigmatism", "Normal"],
                    patch_artist=True,
                    notch=True,
                    showmeans=True)
        plt.title(f"Relative AUC for '{char}' (Naive Word-Frame Alignment)")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

    
    with open("per_word_auc_values.pkl", "wb") as f:
        pickle.dump(per_word_aucAttention2, f)

    return auc_values, per_word_aucAttention2

def area_under_curve_webmouse(words_segments, processor, model,phones_segments):
    """
    1) Get the full STT logits (heatmap) for each audio.
    2) Naively divide the time-frames among characters in the word.
    3) Only extract the portion for characters s, x, or z .
    4) Compute area under the curve (sum of probabilities) in that subrange.
    """

    # Dictionary for aggregated results
    auc_values = {
        "S": {"normal": [], "sigmatism": []},
        "X": {"normal": [], "sigmatism": []},
        "Z": {"normal": [], "sigmatism": []}
    }
    phone_heatmaps = []

    #for segment in tqdm(words_segments, desc="Processing words"):
    for segment in words_segments:

        audio = segment.audio_data
        if audio.size <= 2048:
            continue

        label = segment.label_path  # "normal" or "sigmatism"
        word = segment.label.lower() # Word for filtering relevant characters (assumes lowercase)
        corrected_text = word.replace("ÃŸ", "s").replace("Ã¤", "ä").replace("Ã¼", "ü").replace("Ã¶", "ö")
        phones_in_word = find_pairs(segment, phones_segments)
        # --- 1) Compute the full logits/probs (the "heatmap") ---
        inputs = processor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt", 
            padding=True
        )
        with torch.no_grad():
            logits = model(
                inputs.input_values, 
                attention_mask=inputs.attention_mask
            ).logits
        
        # logits shape: [time_steps, vocab_size]
        probs = F.softmax(logits[0], dim=-1)
        time_steps = probs.shape[0]


        """Attention1-----------------------------------------------------------------------------^
        """
        scaling=16000/44100
        segment_duration_s = (segment.end_time - segment.start_time)
        seconds_per_frame = (segment_duration_s / time_steps)
        print("duration: ",segment_duration_s,seconds_per_frame)
        fig = plt.figure(figsize=(12, 8))
        # Plot the audio waveform
        plt.imshow(logits[0].detach().cpu().numpy().T, aspect='auto', cmap='plasma', origin='lower')
        vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))
        num_vocab = len(vocab_tokens)
        plt.yticks(
            ticks=np.arange(num_vocab),
            labels=vocab_tokens,
            fontsize=6  # might need to reduce font size if it's a large vocab
        )
        plt.title(f"{word} Logits Heatmap with {label}")
        plt.xlabel("Time Steps")
        plt.ylabel("Vocab Index")
        plt.colorbar(label="Logit Value")

        print(np.shape(phones_in_word))
        for phone in phones_in_word:
            print(phone.start_time,phone.end_time, phone.sample_rate,phone.label,phone.label_path,phone.path )
            # phone.start_time / phone.end_time are presumably absolute times
            # relative to the same reference as segment.start_time
            phone_start_sec = (phone.start_time - segment.start_time)*scaling
            phone_end_sec   = (phone.end_time   - segment.start_time)*scaling
            print(phone_start_sec,phone_end_sec)
            # Convert to frame indices
            frame_start = int(phone_start_sec / seconds_per_frame) if seconds_per_frame > 0 else 0
            frame_end   = int(phone_end_sec   / seconds_per_frame) if seconds_per_frame > 0 else 0
            print(frame_start,frame_end)
            if(frame_start ==  frame_end):  
                frame_end = frame_start + 1
            # Make sure it's within bounds
            frame_start = max(0, min(frame_start, time_steps))
            frame_end   = max(0, min(frame_end, time_steps))
            if frame_end <= frame_start:
                continue  # no frames

            plt.axvline(x=frame_start, color='green', linestyle='--', label='Phone Start')
            plt.axvline(x=frame_end, color='red', linestyle='--', label='Phone End')

            # Extract the slice from the full probability "heatmap"
            # shape: [ (frame_end - frame_start), vocab_size ]
            phone_probs_slice = probs[frame_start:frame_end, :].cpu().numpy()
            print("Start: ",frame_start)
            print("END:",frame_end)
            print(phone.label)
            char = phone.label
            if char == "ts":
                char = "s"

            token_id = processor.tokenizer.convert_tokens_to_ids(char)
            if token_id is None:
                # If not found in vocabulary, skip
                continue
            sub_probs = probs[frame_start:frame_end, token_id].cpu().numpy()
            #Compute AUC as the sum of probabilities in that range ---
            auc_char = float(np.sum(sub_probs))  # convert to float for cleanliness

            #  Store results in aggregated dictionary ---
            if label == "normal":
                auc_values[char.upper()]["normal"].append(auc_char)
            elif label == "sigmatism":
                auc_values[char.upper()]["sigmatism"].append(auc_char)
            else:
                # If you have other labels, handle them here
                pass
            # Store
            phone_heatmaps.append({
                "word": word,
                "label": label,
                "character": char,
                "start_frame": frame_start,
                "end_frame": frame_end,
                "auc": auc_char,
                "sub_probs_all_vocab": phone_probs_slice
            })
        plt.legend()
        plt.tight_layout()
        plt.show()
        """Attention 1 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        -----------------------------------------------------------------------------
        """

    # --- 5) Plot aggregated results ---
    # For each relevant char, show boxplot
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]

        if not (normal_aucs or sigmatism_aucs):
            continue  # skip if there's no data for this char

        plt.figure(figsize=(8, 6))
        plt.boxplot([sigmatism_aucs, normal_aucs],
                    labels=["Sigmatism", "Normal"],
                    patch_artist=True,
                    notch=True,
                    showmeans=True)
        plt.title(f"Relative AUC for '{char}' (Naive Word-Frame Alignment)")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

    
    with open("per_word_auc_values.pkl", "wb") as f:
        pickle.dump(phone_heatmaps, f)

    return auc_values, phone_heatmaps

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
    loader = AudioDataLoader()
    words_segments = loader.load_segments_from_pickle("data_lists\words_without_normalization_16kHz.pkl")
    phones = loader.load_segments_from_pickle("data_lists\phone_without_normalization_16kHz.pkl")
    interfere_segments(words_segments)
    #area_under_curve_webmouse(words_segments,processor,model,phones)
    #area_under_curve_relative(words_segments,processor,model)
    #area_under_curve(words_segments)
    #only_take_s(words_segments)


