This repository contains code and tabular datasets for multimodal Multimodal Fusion of Visual and Auditory Biomarkers: An Epoch-Wise Stochastic Modality Masking Framework for Stroke Detection Using a CNN–GRU Network Project.

Project Workflow📖
Feature Extraction from facial images and speech audio -> XAI-Driven Biomarker Identification to highlight the most discriminative features -> Stochastic Modality Masking during training to enhance robustness of 
our Novel Multimodal CNN-GRU Architecture with the the most discriminative features.

📂Repository Structure

💻Code Files	:
1.Facial image to Landmark.py
2.Audio Speech to MFCC.ipynb
3.Top Facial Image Landmark Region Selection.ipynb
4.Top Audio Speech MFCC Feature Selection.ipynb
5.Labelwise Alignment for both modalities.ipynb
6.0.Trainning with epoch wise stochastic modality masking (Cnn for both modalities with MLP fusion).ipynb
6.1.Trainning with epoch wise stochastic modality masking(Cnn-BiLSTM for both modalities with MLP fusion).ipynb
6.2.Trainning with epoch wise stochastic modality masking(Cnn-GRU for both modalities with MLP fusion)).ipynb
6.3.Trainning with epoch wise stochastic modality masking(Cnn-LSTMfor both modalities with MLP fusion)).ipynb
6.4.Trainning with epoch wise stochastic modality masking(Cnn-Lstm facial modality and Cnn-Transformer for audio modality with MLP fusion)).ipynb
6.5.Trainning with epoch wise stochastic modality masking(Cnn-Transformer for both modalities with MLP fusion)).ipynb
6.6.Trainning with epoch wise stochastic modality masking(Dnn-LSTM for both modalities with MLP fusion)).ipynb
6.7.Trainning with epoch wise stochastic modality masking(LSTM for both modalities with MLP fusion)).ipynb
7.Hyperparameter Tunning (LSTM for both modalities with MLP Model fusion)).ipynb
8.Trainning with epoch wise stochastic modality masking(Tunned Cnn-GRU for both modalities with Tunned MLP fusion).ipynb
9.HDBSCAN for model interpretetion.ipynb

⚠️ Note: To run adjust the input paths inside the scripts before running.

📊Tabular Datasets  :
Four preprocessed tabular datasets are included for quick experimentation.
📂ALL_Audio_MFCC_Datadet.csv
📂ALL_Facial_Landmark_Dataset.csv
📂Top_Audio_MFCC_Features.csv
📂Top_Facial_Landmark_Features.csv


📥External Raw Data (⚠️Note : Required for Code 1 & 2)
To run the image and audio feature extraction codes, please download the raw datasets from the following sources and adjust the input paths inside the scripts:

🖼️Facial Image Dataset:
Facial Droop & Paralysis Dataset — Kaitav Mehta, Kaggle, 2019.
Facial Expression Dataset (FER2013) — Astra Szabó, Kaggle, 2020.
🎤Audio Speech Dataset: 
Dysarthria and Non-Dysarthria Speech Dataset — Pooja G., 2023.
TORGO Database (Dysarthric Speech) — Frank Rudzicz et al., 2012.


