This repository contains code and tabular datasets for **Multimodal Fusion of Visual and Auditory Biomarkers: An Epoch-Wise Stochastic Modality Masking Framework for Stroke Detection Using a CNN–GRU Network Project**.

---

## 📖 Project Workflow

Feature Extraction from facial images and speech audio
→ XAI-Driven Biomarker Identification to highlight the most discriminative features
→ Stochastic Modality Masking during training to enhance robustness
→ Our Novel Multimodal CNN-GRU Architecture with the most discriminative features.

![Methodology](https://github.com/user-attachments/assets/fc3cfbe4-b61a-430e-97e5-23c2d542a3cb)

---

⚠️ **Note:** To run, adjust the input paths inside the scripts before execution.

### 📂 Included Tabular Datasets

* ALL_Audio_MFCC_Datadet.csv
* ALL_Facial_Landmark_Dataset.csv
* Top_Audio_MFCC_Features.csv
* Top_Facial_Landmark_Features.csv

---

## 📥 External Raw Data (⚠️ Required for Code 1 & 2)

To run the image and audio feature extraction codes, please download the raw datasets and adjust the input paths:

🖼️ **Facial Image Datasets**

* Facial Droop & Paralysis Dataset — Kaitav Mehta, Kaggle, 2019
* Facial Expression Dataset (FER2013) — Astra Szabó, Kaggle, 2020

🎤 **Audio Speech Datasets**

* Dysarthria and Non-Dysarthria Speech Dataset — Pooja G., 2023
* TORGO Database (Dysarthric Speech) — Frank Rudzicz et al., 2012
