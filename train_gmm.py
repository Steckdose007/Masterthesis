import numpy as np
import librosa
import scipy.fft
import matplotlib.pyplot as plt
from audiodataloader import AudioDataLoader
from sklearn.mixture import GaussianMixture

# Compute MFCCs, Delta MFCCs, and Delta-Delta MFCCs
def compute_mfcc_features(signal, sample_rate, n_mfcc=12, n_mels=22, frame_size=25.6e-3, hop_size=10e-3, n_fft=2048):
    # Convert frame and hop size from seconds to samples
    frame_length = int(frame_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    
    # Compute the static MFCCs using librosa's mfcc function
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, 
                                 n_fft=n_fft, hop_length=hop_length, win_length=frame_length, n_mels=n_mels)
    
    # Compute the first-order difference (Delta MFCCs) using a 5-frame window
    mfcc_delta = librosa.feature.delta(mfccs, width=5)
    
    # Compute the second-order difference (Delta-Delta MFCCs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=5)
    
    # Concatenate static, delta, and delta-delta features to form a 36-dimensional feature vector per frame
    mfcc_features = np.concatenate([mfccs, mfcc_delta, mfcc_delta2], axis=0)
    
    return mfcc_features


def train_ubm(mfcc_features, n_components=16, max_iter=100, reg_covar=1e-6):
    """
    Train a GMM as the Universal Background Model (UBM).
    
    Parameters:
    - mfcc_features: A numpy array of shape (n_frames, n_features).
    - n_components: The number of Gaussian components in the GMM (UBM).
    - max_iter: Maximum number of iterations for fitting the GMM.
    - reg_covar: Regularization added to the diagonal of covariance matrices to prevent singularities.
    
    Returns:
    - ubm: A trained GaussianMixture model (UBM).
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', 
                          max_iter=max_iter, random_state=42, reg_covar=reg_covar)
    
    # Fit the GMM on the MFCC features
    gmm.fit(mfcc_features)
    
    return gmm

def adapt_ubm(ubm, mfcc_features, max_iter=100, reg_covar=1e-6):
    """
    Adapt the UBM to new MFCC features using a simplified version of MAP adaptation.
    
    Parameters:
    - ubm: A pre-trained Universal Background Model (UBM).
    - mfcc_features: The MFCC features of the new utterance or speaker.
    - max_iter: Maximum number of iterations for fitting the adapted GMM.
    
    Returns:
    - adapted_gmm: The adapted GaussianMixture model.
    """
    adapted_gmm = GaussianMixture(n_components=ubm.n_components, covariance_type='diag', 
                                  max_iter=max_iter, random_state=42, reg_covar=reg_covar, 
                                  means_init=ubm.means_, precisions_init=ubm.precisions_)
    
    # Fit the new GMM to the MFCC features (MAP adaptation approximation)
    adapted_gmm.fit(mfcc_features)
    
    return adapted_gmm

def extract_supervector(gmm):
    """
    Extract the supervector from the adapted GMM by concatenating its parameters.
    
    Parameters:
    - gmm: The adapted GaussianMixture model.
    
    Returns:
    - supervector: A flattened numpy array containing the concatenated means, covariances, and weights.
    """
    means = gmm.means_.flatten()  # Mean vectors of the Gaussian components
    covariances = gmm.covariances_.flatten()  # Diagonal covariance elements of the Gaussian components
    weights = gmm.weights_.flatten()  # Mixture weights of the Gaussian components
    
    # Concatenate the means, covariances, and weights into a single supervector
    supervector = np.concatenate([means, covariances, weights])
    
    return supervector


if __name__ == "__main__":

    loader = AudioDataLoader(config_file='config.json', word_data= True, phone_data= False, sentence_data= True)    
    words_segments = loader.create_dataclass_words()
    mfcc_list = []
    for word in words_segments:
        signal = word.audio_data
        sample_rate = word.sample_rate
        # Compute 12 static MFCCs, 24 dynamic (delta and delta-delta) MFCCs, using 22 Mel filters
        mfcc = compute_mfcc_features(signal, sample_rate)
        #Transpose to get it like that: (n_components, n_features) for the covarianve_type: diag
        mfcc_list.append(np.transpose(mfcc))

    print(len(mfcc_list),np.shape(mfcc_list[0]))
    # Concatenate all MFCC features into a single matrix So (n_frames,36 features)
    mfcc_features = np.concatenate(mfcc_list, axis=0)
    print(np.shape(mfcc_features))
    print("Training UBM...")
    ubm = train_ubm(mfcc_features)
    print("Training finished!")

    # Step 2: Adapt the UBM for each word
    print("Adapting UBM for each word...")
    supervectors = []
    for word in words_segments:
        signal = word.audio_data
        mfcc = compute_mfcc_features(signal, word.sample_rate)
        mfcc = np.transpose(mfcc)  # Shape it to (n_frames, n_features)

        # Adapt the UBM to this word
        adapted_gmm = adapt_ubm(ubm, mfcc)
        
        # Step 3: Extract the supervector
        supervector = extract_supervector(adapted_gmm)
        supervectors.append(supervector)
    
    # supervectors now contains the supervectors for each word
    print(f"Extracted {len(supervectors)} supervectors.")

