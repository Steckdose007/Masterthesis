import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import glob
import os
from audiodataloader import AudioDataLoader, AudioSegment, split_list_after_speaker
from tqdm import tqdm 
import librosa

# ------------------------------------------------------------------------
# 1. Load model & processor (modify if you have a custom model/vocab)
# ------------------------------------------------------------------------
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

# ------------------------------------------------------------------------
# 2. Register hooks ONCE
# ------------------------------------------------------------------------
cnn_activations = {}

def hook_cnn(module, input_, output_):
    # output_ is typically shape: (batch_size, channels=512, time_steps)
    cnn_activations["last_cnn"] = output_

cnn_handle = model.wav2vec2.feature_extractor.register_forward_hook(hook_cnn)

transformer_activations = {}

def hook_transformer(module, input_, output_):
    # output_ can be a tuple; usually the hidden states are output_[0]
    # shape: (batch_size, time_steps, hidden_dim=768) for 'base' model
    #print("output_[0]", output_[0].shape)
    transformer_activations["last_transformer"] = output_[0]

transformer_handle = model.wav2vec2.encoder.layers[-1].register_forward_hook(hook_transformer)


# ------------------------------------------------------------------------
# 3. Define a helper function to process a single audio file
# ------------------------------------------------------------------------
def process_audio_file(audio):
    """Returns a list of dicts: [{symbol, transformer_vec, cnn_vec}, ...] for non-blank frames."""
    #audio_data = audio.audio_data
    
   
    
    # 3.2. Tokenize
    input_values = processor(
        audio, return_tensors="pt", sampling_rate=16000, padding=True
    ).input_values
    
    # 3.3. Forward pass
    with torch.no_grad():
        outputs = model(input_values)
    
    # 3.4. Get logits and predicted IDs
    # outputs.logits => (batch_size=1, time_steps, vocab_size)
    logits = outputs.logits.squeeze(0)             # (time_steps, vocab_size)
    predicted_ids = torch.argmax(logits, dim=-1)   # (time_steps,)
    # Check if predicted_ids is entirely zeros
    if torch.all(predicted_ids == 0):
        return None
    #print("predicted_ids: ",predicted_ids)
    #print(audio.label,audio.label_path)
    #print(predicted_ids)
    #print("logits", logits.shape)
    
    # 3.5. Identify non-blank frames
    # For "facebook/wav2vec2-base-960h", blank is typically ID=0. 
    # Check your vocab if it's different.
    non_blank_timesteps = (predicted_ids != 0).nonzero(as_tuple=True)[0]
    #print("non_blank_timesteps  ",non_blank_timesteps)
    # 3.6. Collect hidden activations for those frames
    hidden_info = []
    for t in non_blank_timesteps:
        symbol_id = predicted_ids[t].item()
        # Convert ID -> label. If you're using phones, use your own phone vocab map
        symbol = processor.tokenizer.decode([symbol_id])  # e.g. "A", "B", etc.
        
        # shape of last_transformer: (batch=1, time_steps, hidden_dim=768)
        transformer_vec = transformer_activations["last_transformer"][0, t, :].cpu().numpy()
        
        # shape of last_cnn: (batch=1, channels=512, time_steps)
        cnn_vec = cnn_activations["last_cnn"][0, :, t].cpu().numpy()
        #print("CNN activations shape:", cnn_activations["last_cnn"].shape)
        #print("Transformer activations shape:", transformer_activations["last_transformer"].shape)

        hidden_info.append({
            "symbol": symbol,
            "transformer_vec": transformer_vec,
            "cnn_vec": cnn_vec
        })
    return hidden_info


# ------------------------------------------------------------------------
# 4. Loop over multiple audio files
# ------------------------------------------------------------------------
loader = AudioDataLoader()
words_segments = loader.load_segments_from_pickle("data_lists\words_without_normalization_16kHz.pkl")
#words_segments = loader.load_segments_from_pickle("old_lists/sentences__16kHz.pkl")
segments = words_segments[:100]
combined_audio = np.concatenate([segment.audio_data for segment in segments])
plt.figure(figsize=(10, 7))
#audio, sr = librosa.load("Data/normal/Audio_1.wav", sr=None) 
#for i in tqdm(range(1), desc="Processing words"):
file_hidden_info = process_audio_file(combined_audio)
# if file_hidden_info == None:
#     continue
# ------------------------------------------------------------------------
# 5. Now we have a big list of activations for all utterances
#    Let's do PCA on the final transformer layer as an example
# ------------------------------------------------------------------------
transformer_matrix = np.array([x["transformer_vec"] for x in file_hidden_info])
symbols = [x["symbol"] for x in file_hidden_info]
#print("transformer matrix: ",np.shape(transformer_matrix))
pca = PCA(n_components=2)
transformer_pca = pca.fit_transform(transformer_matrix)  # (num_frames_across_all_utts, 2)
unique_symbols = list(set(symbols))
desired_symbols = {"s", "z", "x", "n","e"}
for sym in desired_symbols:
    idxs = [j for j, s in enumerate(symbols) if s == sym]
    if idxs:  # Check if there are any points for the symbol
        plt.scatter(transformer_pca[idxs, 0], transformer_pca[idxs, 1], label=sym, alpha=0.6)
# for sym in unique_symbols:
#     idxs = [i for i, s in enumerate(symbols) if s == sym]
#     plt.scatter(transformer_pca[idxs, 0], transformer_pca[idxs, 1], label=sym, alpha=0.6)

plt.title("PCA of Final Transformer Layer Activations (All Utterances, Non-blank Frames)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------
# 6. Remove hooks (clean up)
# ------------------------------------------------------------------------
cnn_handle.remove()
transformer_handle.remove()

