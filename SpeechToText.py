import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from audiodataloader import AudioDataLoader, AudioSegment, split_list_after_speaker
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

def interfere_segments(words_segments):
    """
    Visualisiert die Logits und Wahrscheinlichkeiten eines bestimmten Wortsegments
    mithilfe eines STT-Modells (Speech-to-Text). Geplottet wird ein Heatmap der Logits
    und ein Plot der Wahrscheinlichkeit für den Laut 's' über die Zeit.
    """

    index = 5030
    segment = words_segments[index]
    audio = segment.audio_data

    print("Word: ", segment.label)
    print("Label: ", segment.label_path)

    # Vorverarbeitung des Audios
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # STT-Modellvorhersage (Logits)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    print("output", logits.shape)

    # Ausgabe und Visualisierung
    print_outputs(logits, segment.label, segment.label_path)

    # Wahrscheinlichkeiten berechnen
    probs = F.softmax(logits, dim=-1)
    print_outputs(probs, segment.label, segment.label_path)

    # Wahrscheinlichkeit für "s" über die Zeit
    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")
    s_probs = probs[0, :, s_token_id]
    print("Shape of s_probs:", s_probs.shape)

    # Dekodierte Vorhersage (Text)
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    print("-" * 100)
    print("Reference:", segment.label)
    print("Prediction:", predicted_sentences)
    plt.show()

def print_outputs(output, label_word, label_path):
    """
    Erstellt zwei Visualisierungen:
    (1) Ein Heatmap der Logits oder Wahrscheinlichkeiten über Zeit und Vokabular.
    (2) Ein Plot der Wahrscheinlichkeit des Lautes 's' über die Zeit.
    """

    logits = output[0]  # [time_steps, vocab_size]
    logits_np = logits.detach().cpu().numpy()

    # Heatmap der Logits/Wahrscheinlichkeiten
    plt.figure(figsize=(12, 6))
    plt.imshow(logits_np.T, aspect='auto', cmap='plasma', origin='lower')
    vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))
    plt.yticks(ticks=np.arange(len(vocab_tokens)), labels=vocab_tokens, fontsize=6)
    plt.title(f"{label_word} Logits Heatmap with {label_path}")
    plt.xlabel("Time Steps")
    plt.ylabel("Vocab Index")
    plt.colorbar(label="Logit Value")
    plt.tight_layout()

    # Wahrscheinlichkeit des Tokens "s" über die Zeit
    probs = F.softmax(logits, dim=-1)
    s_token_id = processor.tokenizer.convert_tokens_to_ids("s")
    s_probs = probs[:, s_token_id].cpu().numpy()
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
    Berechnet die Fläche unter der Kurve (AUC) für die Wahrscheinlichkeiten der Laute
    's', 'x' und 'z' über die Zeit für jedes Wortsegment. Die Ergebnisse werden
    getrennt für "normal" und "sigmatism" ausgewertet und visualisiert.
    """

    auc_values = {"S": {"normal": [], "sigmatism": []},
                  "X": {"normal": [], "sigmatism": []},
                  "Z": {"normal": [], "sigmatism": []}}
    per_word_auc = []

    for segment in words_segments:
        audio = segment.audio_data
        if audio.size <= 2048:
            continue

        label = segment.label_path
        word = segment.label.lower()
        corrected_text = word.replace("ÃŸ", "s").replace("ãÿ", "s").replace("Ã¤", "ä").replace("ã¼", "ü").replace("Ã¼", "ü").replace("Ã¶", "ö").replace("ã¤", "ä").lower()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        probs = F.softmax(logits[0], dim=-1)

        for char in ["s", "x", "z"]:
            if char in corrected_text:
                token_id = processor.tokenizer.convert_tokens_to_ids(char)
                char_probs = probs[:, token_id].cpu().numpy()
                auc = np.sum(char_probs)
                sum_logits = np.sum(logits[0][:, token_id].cpu().numpy())

                if label == "normal":
                    auc_values[char.upper()]["normal"].append(auc)
                elif label == "sigmatism":
                    auc_values[char.upper()]["sigmatism"].append(auc)

                per_word_auc.append({
                    "label": corrected_text,
                    "label_path": label,
                    "path": segment.path,
                    "auc": float(auc),
                    "heatmap": logits[0],
                    "logtis_AuC": sum_logits
                })

    with open("per_word_auc_values_deivided_by_nums.pkl", "wb") as f:
        pickle.dump(per_word_auc, f)

    # Visualisierung
    for char in ["S", "X", "Z"]:
        plt.figure(figsize=(8, 6))
        plt.boxplot([auc_values[char]["sigmatism"], auc_values[char]["normal"]],
                    labels=['Sigmatism', 'Normal'], patch_artist=True, notch=True, showmeans=True)
        plt.title(f"Average Area Under Curve (AUC) for '{char}'")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.hist(auc_values[char]["normal"], bins=40, alpha=0.5, label='Normal')
        plt.hist(auc_values[char]["sigmatism"], bins=40, alpha=0.5, label='Sigmatism')
        plt.title('AUC Distributions: Normal vs. Sigmatism')
        plt.xlabel('AUC')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()


def area_under_curve_relative(words_segments, processor, model):
    """
    Computes the relative Area Under Curve (AUC) for characters 's', 'x', and 'z' by:
    1. Getting STT logits for each segment.
    2. Mapping frames uniformly across characters in the word.
    3. Extracting sub-probabilities per character and computing AUC.
    4. Visualizing results and saving per-character segment-level statistics.
    """
    auc_values = {
        "S": {"normal": [], "sigmatism": []},
        "X": {"normal": [], "sigmatism": []},
        "Z": {"normal": [], "sigmatism": []}
    }
    per_word_aucAttention2 = []
    target_chars = ["s", "x", "z"]

    for segment in words_segments:
        audio = segment.audio_data
        if audio.size <= 2048:
            continue

        label = segment.label_path
        word = segment.label.lower()
        corrected_text = word.replace("ÃŸ", "ss").replace("ãÿ", "ss").replace("Ã¤", "ä").replace("ã¼", "ü").replace("Ã¼", "ü").replace("Ã¶", "ö").replace("ã¤", "ä").lower()

        # 1) Get STT logits and convert to probabilities
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        probs = F.softmax(logits[0], dim=-1)
        time_steps = probs.shape[0]

        # Plot full heatmap for this segment
        plt.figure(figsize=(12, 8))
        plt.imshow(logits[0].detach().cpu().numpy().T, aspect='auto', cmap='plasma', origin='lower')
        vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))
        plt.yticks(np.arange(len(vocab_tokens)), labels=vocab_tokens, fontsize=6)
        plt.title(f"{corrected_text} Logits Heatmap with {label}")
        plt.xlabel("Time Steps")
        plt.ylabel("Vocab Index")
        plt.colorbar(label="Logit Value")

        # 2) Divide frames uniformly across characters
        num_chars = len(corrected_text)
        if num_chars == 0:
            continue
        frames_per_char = time_steps / num_chars

        for idx, char in enumerate(corrected_text):
            if char not in target_chars:
                continue
            # Skip compound consonants like 'sch' or 'st'
            if corrected_text[idx:idx+3] == "sch" or corrected_text[idx:idx+2] == "st":
                continue

            # Calculate frame range for this character
            start_frame = int(np.floor(idx * frames_per_char))
            end_frame = int(np.floor((idx + 1) * frames_per_char))
            start_frame = max(0, start_frame)
            end_frame = min(time_steps, end_frame)

            if start_frame >= end_frame:
                continue

            # Plot indicators on the heatmap
            plt.axvline(x=start_frame, color='green', linestyle='--', label='Char Start')
            plt.axvline(x=end_frame, color='red', linestyle='--', label='Char End')

            # Extract sub-probabilities for this char
            token_id = processor.tokenizer.convert_tokens_to_ids(char)
            if token_id is None:
                continue
            sub_probs = probs[start_frame:end_frame, token_id].cpu().numpy()
            auc_char = float(np.sum(sub_probs))

            if label == "normal":
                auc_values[char.upper()]["normal"].append(auc_char)
            elif label == "sigmatism":
                auc_values[char.upper()]["sigmatism"].append(auc_char)

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

    # 5) Final plotting
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]
        if not (normal_aucs or sigmatism_aucs):
            continue

        plt.figure(figsize=(8, 6))
        plt.boxplot([sigmatism_aucs, normal_aucs], labels=["Sigmatism", "Normal"],
                    patch_artist=True, notch=True, showmeans=True)
        plt.title(f"Relative AUC for '{char}' (Naive Frame Mapping)")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

    with open("per_word_auc_values.pkl", "wb") as f:
        pickle.dump(per_word_aucAttention2, f)

    return auc_values, per_word_aucAttention2


def area_under_curve_webmouse(words_segments, processor, model, phones_segments):
    """
    Computes the AUC for phones ('s', 'x', 'z') using phone-level time alignment.
    Steps:
    1. Get STT logits (logit heatmap) for each word audio.
    2. Use phonetic alignment info (phones_segments) to determine frame ranges.
    3. Compute AUC over those ranges for each relevant phone.
    4. Store and visualize results (boxplots).
    """

    auc_values = {
        "S": {"normal": [], "sigmatism": []},
        "X": {"normal": [], "sigmatism": []},
        "Z": {"normal": [], "sigmatism": []}
    }
    phone_heatmaps = []

    for segment in words_segments:
        audio = segment.audio_data
        if audio.size <= 2048:
            continue

        label = segment.label_path
        word = segment.label.lower()
        corrected_text = word.replace("ÃŸ", "s").replace("Ã¤", "ä").replace("Ã¼", "ü").replace("Ã¶", "ö")

        # Find matching phone-level segments
        phones_in_word = find_pairs(segment, phones_segments)

        # Run STT model
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        probs = F.softmax(logits[0], dim=-1)
        time_steps = probs.shape[0]

        # Compute timing scaling between segment and STT output
        scaling = 16000 / 44100  # audio was resampled?
        segment_duration_s = (segment.end_time - segment.start_time)
        seconds_per_frame = segment_duration_s / time_steps

        # Plot full heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(logits[0].detach().cpu().numpy().T, aspect='auto', cmap='plasma', origin='lower')
        vocab_tokens = processor.tokenizer.convert_ids_to_tokens(range(logits.shape[-1]))
        plt.yticks(np.arange(len(vocab_tokens)), labels=vocab_tokens, fontsize=6)
        plt.title(f"{word} Logits Heatmap with {label}")
        plt.xlabel("Time Steps")
        plt.ylabel("Vocab Index")
        plt.colorbar(label="Logit Value")

        # Loop over phones
        for phone in phones_in_word:
            phone_start_sec = (phone.start_time - segment.start_time) * scaling
            phone_end_sec = (phone.end_time - segment.start_time) * scaling

            frame_start = int(phone_start_sec / seconds_per_frame) if seconds_per_frame > 0 else 0
            frame_end = int(phone_end_sec / seconds_per_frame) if seconds_per_frame > 0 else 0
            if frame_start == frame_end:
                frame_end = frame_start + 1

            frame_start = max(0, min(frame_start, time_steps))
            frame_end = max(0, min(frame_end, time_steps))
            if frame_end <= frame_start:
                continue

            plt.axvline(x=frame_start, color='green', linestyle='--', label='Phone Start')
            plt.axvline(x=frame_end, color='red', linestyle='--', label='Phone End')

            # Handle special case: "ts" → treat as "s"
            char = phone.label
            if char == "ts":
                char = "s"

            token_id = processor.tokenizer.convert_tokens_to_ids(char)
            if token_id is None:
                continue

            sub_probs = probs[frame_start:frame_end, token_id].cpu().numpy()
            auc_char = float(np.sum(sub_probs))

            if label == "normal":
                auc_values[char.upper()]["normal"].append(auc_char)
            elif label == "sigmatism":
                auc_values[char.upper()]["sigmatism"].append(auc_char)

            phone_heatmaps.append({
                "word": word,
                "label": label,
                "character": char,
                "start_frame": frame_start,
                "end_frame": frame_end,
                "auc": auc_char,
                "sub_probs_all_vocab": probs[frame_start:frame_end, :].cpu().numpy()
            })

        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot per-character comparison
    for char in ["S", "X", "Z"]:
        normal_aucs = auc_values[char]["normal"]
        sigmatism_aucs = auc_values[char]["sigmatism"]
        if not (normal_aucs or sigmatism_aucs):
            continue

        plt.figure(figsize=(8, 6))
        plt.boxplot([sigmatism_aucs, normal_aucs], labels=["Sigmatism", "Normal"],
                    patch_artist=True, notch=True, showmeans=True)
        plt.title(f"Relative AUC for '{char}' (Forced Alignment)")
        plt.ylabel("AUC")
        plt.tight_layout()
        plt.show()

    with open("per_word_auc_values.pkl", "wb") as f:
        pickle.dump(phone_heatmaps, f)

    return auc_values, phone_heatmaps

def only_take_s(words_segments):
    """
    Computes the probability of 's' only at those time steps where it is the top prediction.
    Useful for analyzing how confident the model is in detecting 's' where it's most likely.
    """

    probabilities_normal = []
    probabilities_sigmatism = []

    for segment in tqdm(words_segments, desc="Processing words"):
        audio = segment.audio_data
        label = segment.label_path

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        logits = outputs[0]  # [time_steps, vocab_size]
        probs = F.softmax(logits, dim=-1)

        s_token_id = processor.tokenizer.convert_tokens_to_ids("s")

        predicted_ids = torch.argmax(probs, dim=-1)
        s_time_steps = torch.where(predicted_ids == s_token_id)[0]

        if len(s_time_steps) > 0:
            s_probs = probs[s_time_steps, s_token_id].cpu().numpy()
            avg_s_prob = np.mean(s_probs)
        else:
            avg_s_prob = 0.0

        if label == "normal":
            probabilities_normal.append(avg_s_prob)
        elif label == "sigmatism":
            probabilities_sigmatism.append(avg_s_prob)
        else:
            raise ValueError(f"Unexpected label: {label}")

    # Plot
    data = [probabilities_normal, probabilities_sigmatism]
    labels = ["Normal", "Sigmatism"]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red"))

    plt.title("Probabilities of 'S' When Predicted as Most Likely")
    plt.ylabel("Probability of 'S'")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.show()

def make_heatmap_list(words_segments):
    # To store per-word results
    per_word_auc = []
    dataset_length = len(words_segments)

    # 2. Loop through words
    for i in tqdm(range(dataset_length), desc="Processing words"):
    #for i in range(dataset_length):

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
        
        # -- Store per-word AUC --
        per_word_auc.append({
            "label": corrected_text,  #word for example "sonne"
            "label_path": label, # sigmatism or normal
            "path": segment.path, # which file it is from
            "heatmap": logits[0],
        })

    #save
    with open("STT_heatmap_list.pkl", "wb") as f:
        pickle.dump(per_word_auc, f)


if __name__ == "__main__":

    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    loader = AudioDataLoader()
    words_segments = loader.load_segments_from_pickle("data_lists\words_without_normalization_16kHz.pkl")
    phones = loader.load_segments_from_pickle("data_lists\phone_without_normalization_16kHz.pkl")
    make_heatmap_list(words_segments)
    #interfere_segments(words_segments)
    #area_under_curve_webmouse(words_segments,processor,model,phones)
    #area_under_curve_relative(words_segments,processor,model)
    #area_under_curve(words_segments)
    #only_take_s(words_segments)


