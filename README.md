# Unrolled Creative Adversarial Network for Generating Novel Musical Pieces  

## Overview  

This repository provides an implementation of the **Unrolled Creative Adversarial Network (CAN)**, designed to generate novel musical compositions. By leveraging deep learning techniques, CAN produces innovative and contextually meaningful music. The model incorporates multiple layers and iterative training processes to generate original pieces across various musical styles.  

**Paper Link:** [arXiv:2501.00452](https://arxiv.org/abs/2501.00452)  

## Features  

- **Generative Model:** Utilizes a combination of Generative Adversarial Networks (GANs) and creative unrolling techniques to enhance musical creativity.  
- **Novel Music Generation:** Produces compositions that are unique and distinct from existing works.  
- **Multi-Genre Adaptability:** Supports training on various musical genres, enabling the generation of style-specific pieces.  
- **Interactive Interface:** Allows real-time modifications of style, genre, and mood for user-customized music generation.  

## Requirements  

- Python 3.7+  
- TensorFlow  
- NumPy  

## Reproducing the Results  

### Training the Unrolled CAN Model  
To train the **Unrolled CAN model**, execute the `GAN_reconstructed.ipynb` file located in the `CAN_models` folder. Sample data is available in the `classical_data_midi` directory. The core model configurations are in `training.py` and `model.py`. MIDI datasets can be preprocessed into images using `data_pre_processing.ipynb`.  

### Training the DCGAN Model  
For training the **DCGAN model**, navigate to the `GAN_models` folder and execute `GAN-Copy2.ipynb`. Ensure the correct dataset path is specified before running the script. Sample outputs for both models are included in their respective directories.  

### Evaluating Novelty Scores  
To evaluate **novelty scores**, run the `novelty_score.ipynb` script located in the home directory.  

## Datasets  

The datasets used in this study are publicly available:  
- [Jazz MIDI Dataset](https://www.kaggle.com/saikayala/jazz-ml-ready-midi)  
- [Classical MIDI Dataset](https://www.kaggle.com/soumikrakshit/classical-music-midi)  

## Repository Structure  

Unrolled_CAN/
│
├── CAN_models/                   # Contains CAN model scripts  
│   ├── GAN_reconstructed.ipynb    # Main training script for Unrolled CAN  
│   ├── training.py                # Training configurations  
│   ├── model.py                   # Model architecture  
│
├── GAN_models/                    # Contains DCGAN model scripts  
│   ├── GAN-Copy2.ipynb             # DCGAN training script  
│
├── classical_data_midi/            # Sample MIDI dataset  
│
├── datasets/                       # Additional dataset directory  
│
├── results/                        # Stores generated music and logs  
│
├── novelty_score.ipynb             # Novelty evaluation script  
├── data_pre_processing.ipynb       # Preprocessing script for MIDI datasets  
├── requirements.txt                # Required dependencies  
├── README.md                       # Project documentation  


## Citation  

If you use this code, please cite:  

@article{unrolled_can_nag, author = {Pratik Nag}, title = {Unrolled Creative Adversarial Network Generating Novel Musical Pieces}, journal = {arXiv preprint}, year = {2025}, url = {https://arxiv.org/abs/2501.00452} }
