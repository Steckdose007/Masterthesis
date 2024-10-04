import numpy as np
import librosa
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
    mfcc_delta = librosa.feature.delta(mfccs, width=3)
    
    # Compute the second-order difference (Delta-Delta MFCCs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=3)
    
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

def compute_posterior_probs(gmm, mfcc_features):
    """
    Compute the posterior probabilities (responsibilities) for each Gaussian component 
    in the UBM for the given MFCC features.

    Parameters:
    - gmm: The trained GMM (UBM).
    - mfcc_features: A numpy array of shape (n_frames, n_features).

    Returns:
    - responsibilities: A numpy array of shape (n_frames, n_components), which contains 
      the posterior probabilities (responsibilities) for each frame and Gaussian component.
    """
    log_prob_norm, responsibilities = gmm._estimate_log_prob_resp(mfcc_features)
    return responsibilities

def update_means(ubm, responsibilities, mfcc_features, relevance_factor):
    """
    Update the means of the UBM components using MAP adaptation.

    Parameters:
    - ubm: The Universal Background Model (GMM).
    - responsibilities: The posterior probabilities for each Gaussian component (n_frames, n_components).
    - mfcc_features: The MFCC features (n_frames, n_features).
    - relevance_factor: The MAP relevance factor (controls the influence of the UBM means vs. new data).

    Returns:
    - adapted_means: The adapted means of the GMM components.
    """
    # Calculate effective number of data points for each component (N_k)
    N_k = np.sum(responsibilities, axis=0)  # Shape: (n_components,)

    # Calculate the new data means (weighted by responsibilities)
    weighted_sum = np.dot(responsibilities.T, mfcc_features)  # Shape: (n_components, n_features)

    # Update the means using MAP formula
    adapted_means = (N_k[:, np.newaxis] * weighted_sum + relevance_factor * ubm.means_) / (N_k[:, np.newaxis] + relevance_factor)

    return adapted_means

def update_covariances(ubm, responsibilities, mfcc_features, adapted_means, relevance_factor):
    """
    Update the covariances of the UBM components using MAP adaptation.

    Parameters:
    - ubm: The Universal Background Model (GMM).
    - responsibilities: The posterior probabilities for each Gaussian component (n_frames, n_components).
    - mfcc_features: The MFCC features (n_frames, n_features).
    - adapted_means: The adapted means of the GMM components (from the update_means step).
    - relevance_factor: The MAP relevance factor (controls the influence of the UBM covariances vs. new data).

    Returns:
    - adapted_covariances: The adapted covariances of the GMM components.
    """
    # Calculate effective number of data points for each component (N_k)
    N_k = np.sum(responsibilities, axis=0)  # Shape: (n_components,)

    # Calculate the weighted sum of square deviations from the adapted means
    diff = mfcc_features[:, np.newaxis, :] - adapted_means  # Shape: (n_frames, n_components, n_features)
    weighted_diff = responsibilities[:, :, np.newaxis] * (diff ** 2)

    # Compute new covariances based on weighted differences
    adapted_covariances = (np.sum(weighted_diff, axis=0) + relevance_factor * ubm.covariances_) / (N_k[:, np.newaxis] + relevance_factor)

    return adapted_covariances

def update_weights(ubm, responsibilities, relevance_factor):
    """
    Update the weights of the UBM components using MAP adaptation.

    Parameters:
    - ubm: The Universal Background Model (GMM).
    - responsibilities: The posterior probabilities for each Gaussian component (n_frames, n_components).
    - relevance_factor: The MAP relevance factor (controls the influence of the UBM weights vs. new data).

    Returns:
    - adapted_weights: The adapted weights of the GMM components.
    """
    # Calculate effective number of data points for each component (N_k)
    N_k = np.sum(responsibilities, axis=0)  # Shape: (n_components,)

    # Update the weights using MAP formula
    adapted_weights = (N_k + relevance_factor * ubm.weights_) / (np.sum(N_k) + relevance_factor)

    return adapted_weights

def adapt_ubm_map(ubm, mfcc_features, relevance_factor=16):
    """
    Perform MAP adaptation of the UBM on new MFCC features.

    Parameters:
    - ubm: The Universal Background Model (GMM).
    - mfcc_features: A numpy array of MFCC features (n_frames, n_features).
    - relevance_factor: The MAP relevance factor (typically between 10 and 20).

    Returns:
    - adapted_gmm: A new GMM with adapted parameters.
    """
    # Compute the posterior probabilities (responsibilities)
    responsibilities = compute_posterior_probs(ubm, mfcc_features)

    # Update GMM parameters (means, covariances, and weights) using MAP adaptation
    adapted_means = update_means(ubm, responsibilities, mfcc_features, relevance_factor)
    adapted_covariances = update_covariances(ubm, responsibilities, mfcc_features, adapted_means, relevance_factor)
    adapted_weights = update_weights(ubm, responsibilities, relevance_factor)

    # Create the adapted GMM
    adapted_gmm = GaussianMixture(n_components=ubm.n_components, covariance_type='diag')
    adapted_gmm.means_ = adapted_means
    adapted_gmm.covariances_ = adapted_covariances
    adapted_gmm.weights_ = adapted_weights
    adapted_gmm.precisions_cholesky_ = 1 / np.sqrt(adapted_covariances)

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
    weights = gmm.weights_  # Mixture weights of the Gaussian components
    
    # Concatenate the means, covariances, and weights into a single supervector
    supervector = np.concatenate([means, covariances, weights])
    simplified_supervector = weights
    
    return supervector, simplified_supervector


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
    ubm = train_ubm(mfcc_features, n_components=16, max_iter=100, reg_covar=1e-6)
    print("Training finished!")

    # Step 2: Adapt the UBM for each word
    print("Adapting UBM for each word...")
    supervectors = []
    simmplified_supervectors = []
    for word in words_segments:
        signal = word.audio_data
        mfcc = compute_mfcc_features(signal, word.sample_rate)
        mfcc = np.transpose(mfcc)  # Shape it to (n_frames, n_features)

        # Adapt the UBM to this word
        #print("mfcc word adaption shape:", np.shape(mfcc))
        adapted_gmm = adapt_ubm_map(ubm, mfcc)
        
        # Step 3: Extract the supervector
        supervector, simmplified_supervector = extract_supervector(adapted_gmm)
        #print(np.shape(supervector),np.shape(simmplified_supervector))
        supervectors.append(supervector)
        simmplified_supervectors.append(simmplified_supervector)
    
    # supervectors now contains the supervectors for each word
    print(f"Extracted {len(supervectors)} supervectors and {len(simmplified_supervectors)} simplified supervectors")

