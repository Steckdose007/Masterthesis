Masterthesis Repository

This repository contains code and resources related to the master's thesis project focused on speech processing and analysis, particularly concerning sigmatism detection and attention mechanisms in deep learning models.
Repository Structure and File Descriptions


    Data/: Contains datasets and related resources used for training and evaluation.​

    DeeplearningPaper/: Includes implementations and notes from relevant deep learning research papers referenced during the project.​

    graphics/: Contains visual assets such as plots, graphs, and images generated or used in the project.​

    old code/: Archive of previous versions or deprecated scripts from earlier stages of development.​

    Dataloader_fixedlist.py: Script for loading datasets. Used in train_CNN.​py.

    Dataloader_gradcam.py: Handles data loading tailored for Grad-CAM analysis. Used in train_gmm.py​.

    SpeechToText.py: Plots utilizes and shows STT heatmap. Also implements the Bimodal AUC approach.​

    audiodataloader.py: Manages the loading and preprocessing of audio datasets for training and evaluation purposes.​ Takes the Data with hoel wav files and creates the word_lists with the extracted word and the metadata. The list is then used in training.

    config.json: Configuration file storing parameters and settings used across various scripts in the project.​

    cpp.py: Approach to distingquish our two classes with a metric. Here the cpp and FID are implemented

    create_fixed_list.py: Generates a fixed list of data samples for training.

    data_augmentation.py: Contains methods for augmenting audio data, such as adding noise or altering pitch.​

    gradcam.py: Implements the Grad-CAM algorithm for visualizing the regions of input data that influence model predictions.​

    hyperparametertuning.py: Facilitates the search and optimization of hyperparameters using optuna.​

    jobscript.sh: Shell script for submitting jobs to a computing cluster or managing batch processing tasks.​

    model.py: Defines the architecture of the deep learning models employed in the project.​

    paperimplementation.py: Contains code to implement the valentini paper as our baseline.​

    plotting.py: Provides functions for generating visualizations, aiding in data analysis and result interpretation.​

    resample_data.py: Handles the resampling of a whole folder. Use to put them into the STT model.

    train_CNN.py: Script for training Convolutional Neural Networks on the prepared datasets.​

    train_gmm.py: Used by paperimplementation.py.
