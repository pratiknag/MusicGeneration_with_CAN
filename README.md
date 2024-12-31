
# Unrolled Creative Adversarial Network Generating Novel Musical Pieces

## Overview

This project implements the **Unrolled Creative Adversarial Network (CAN)** designed to generate novel musical pieces. CAN uses deep learning to create music that is both innovative and contextually meaningful. The network consists of multiple layers and training iterations, allowing it to produce original compositions in a variety of styles.

## Features

- **Generative Model:** CAN uses a combination of generative adversarial networks (GANs) and creative unrolling techniques to enhance the creativity of generated musical pieces.
- **Novelty in Music:** The network is designed to produce music that is unique and distinctive from existing works.
- **Multi-Style Support:** The CAN model can be trained on various genres, producing music that fits within different musical styles.
- **Interactive Interface:** The model supports real-time input, allowing users to modify the style, genre, or mood of the generated music.

## Requirements

- Python 3.7+
- TensorFlow 
- NumPy

## Reproducing the Results

To train the Unrolled CAN model on the dataset, the file `GAN_reconstructed.ipynb` located in the folder `CAN_models` should be executed. Sample data has been provided in the `classical_data_midi` folder. The files `training.py` and `model.py` contain all the model-related configurations. MIDI datasets can be preprocessed into images using the `data_pre_processing.ipynb` file.

To train the second-best model, the DCGAN, the folder `GAN_models` should be accessed, and the file `GAN-Copy2.ipynb` must be run. Before execution, the proper folder path containing the image representation of the MIDI datasets should be specified in the `GAN-Copy2.ipynb` file. Some generated results for both models have been uploaded to their respective folders.

The `novelty_score.ipynb` file has been utilized to calculate the novelty score, as discussed in [Aljundi et al., 2017](https://www.aclweb.org/anthology/D17-1229). This file is located in the home directory.

The datasets used in this work are publicly available:  
[Jazz MIDI Dataset](https://www.kaggle.com/saikayala/jazz-ml-ready-midi) and [Classical MIDI Dataset](https://www.kaggle.com/soumikrakshit/classical-music-midi).
