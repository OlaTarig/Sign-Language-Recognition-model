#  Arabic Sign Language Recognition — KArSL-502

> Signer-independent Arabic Sign Language Recognition using
> Temporal Convolutional Networks (TCN) trained on the KArSL-502 dataset.
> Achieves **73.9% Top-1** and **92.6% Top-5** accuracy on an unseen signer
> across **252 sign classes**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Details](#model-details)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

This repository contains the full pipeline for Arabic Sign Language
Recognition built as part of a graduation project integrating with
a Flutter-based video meeting application for the deaf community.

The system:
- Extracts hand keypoints from raw video frames using **MediaPipe Holistic**
- Trains a **Temporal Convolutional Network (TCN)** on the KArSL-502 dataset
- Evaluates in a **signer-independent** setting (trained on Signers 1+2,
  tested on unseen Signer 3)
- Exports the final model as **TFLite** for on-device mobile deployment

---

## Results

| Metric      | Full Model (417 classes) | Good Classes Model (252 classes) |
|-------------|--------------------------|----------------------------------|
| Top-1       | 52.7%                    | **73.9%**                        |
| Top-5       | 78.4%                    | **92.6%**                        |
| Top-10      | 84.2%                    | **96.3%**                        |
| Macro F1    | 48.9%                    | **71.7%**                        |
| Weighted F1 | 49.0%                    | **71.9%**                        |

> Evaluated on Signer03 test set — completely unseen during training.

---

## Dataset

**KArSL-502** — King Abdulaziz Arabic Sign Language Dataset

- 502 Arabic sign classes
- 3 professional signers
- ~60,000+ sign recordings
- Distributed as ZIP archives of sequential JPEG frames

```
Download: https://www.kaggle.com/datasets/yousefdotpy/karsl-502
Paper: Sidig et al., "KArSL: Arabic Sign Language Database",
       ACM TALLIP, vol. 20, no. 1, pp. 1-19, 2021.
```

---

## Repository Structure

```
📦 arabic-sign-language-recognition
│
├── 📓 notebooks/
│   ├── Data_Extraction.ipynb  
│   ├── BiLSTM VS TCN.ipynb   
│   ├── Final Model.ipynb   
├── 📄 documentation/
│   └── model_documentation.pdf            # Full model documentation
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

## Pipeline

```
Raw ZIP archives (JPEG frames)
        ↓
MediaPipe Holistic keypoint extraction
        ↓
NumPy arrays  — shape: (T, 63) per hand
        ↓
Wrist-centered normalization
        ↓
Temporal resampling to 48 frames
        ↓
Dataset shape: (N, 48, 126)  float32
        ↓
TCN training — signer-independent split
        ↓
Per-class F1 analysis + class filtering
        ↓
Retrain on 252 good classes
        ↓
Export to TFLite (.tflite)
```

**Data Split:**

| Split      | Data                        | Role           |
|------------|-----------------------------|----------------|
| Train      | Signer01 + Signer02 train   | Seen signers   |
| Validation | Signer03 train              | Unseen signer  |
| Test       | Signer03 test               | Unseen signer  |

---

## Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.13.0
mediapipe==0.10.13
numpy<2
opencv-python
scikit-learn
pandas
matplotlib
seaborn
tqdm
keras-tcn
```

> ⚠️ MediaPipe 0.10.13 requires `protobuf==3.20.3`.
> Use a clean virtual environment to avoid conflicts.

---

## Usage

### 1. Extract Keypoints

Run the extraction notebooks in order for each signer.
Keypoints will be saved as `.npy` files:

```
karsl_work/
├── 01/train/word_folder/lh_keypoints/video.npy   # shape: (T, 63)
├── 01/train/word_folder/rh_keypoints/video.npy   # shape: (T, 63)
└── 01/train/word_folder/pose_keypoints/video.npy # shape: (T, 99)
```

### 2. Build Datasets

Run `05_build_datasets.ipynb` to generate:
```
X_train.npy  — (35075, 48, 126)
X_val.npy    — (17445, 48, 126)
X_test.npy   — (3371,  48, 126)
```

### 3. Train

```python
# Full model — 417 classes
# Run: 06_train_full_model.ipynb

# Good classes model — 252 classes
# Run: 08_train_good_classes.ipynb
```

### 4. Export to TFLite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open("model_good.tflite", "wb") as f:
    f.write(tflite)
```

### 5. Live Demo (OpenCV)

Place these files in the same folder:
```
model_good.tflite
label_map.json
arial.ttf
```

Then run:
```bash
python demo.py
```

Controls:
- `SPACE` — start signing (captures 48 frames)
- `Q` — quit

---

## Model Details

| Property       | Value                          |
|----------------|--------------------------------|
| Architecture   | Temporal Convolutional Network |
| Input shape    | (1, 48, 126)                   |
| Output shape   | (1, 252)                       |
| Features       | Left hand + Right hand keypoints |
| Normalization  | Wrist-centered per hand        |
| Parameters     | 1,600,033                      |
| Model size     | 6.10 MB (.keras)               |
| TFLite size    | ~1.5 MB (quantized)            |
| Training       | Adam, lr=0.001, batch=64       |
| Early stopping | patience=7, monitor=val_acc    |

**Feature extraction:**
```python
# 21 landmarks × 3 coords per hand = 63 features
# Centered on wrist (landmark 0)
lh_adj = landmarks - wrist_position   # (63,)
rh_adj = landmarks - wrist_position   # (63,)
features = concat(lh_adj, rh_adj)     # (126,)
```

---

## Documentation

Full documentation covering dataset description, EDA, preprocessing,
model architecture, training procedure, and evaluation results is
available in:

```
documentation/model_documentation.pdf
```

Topics covered:
- Problem definition and feature representation justification
- KArSL-502 dataset structure and statistics
- Exploratory data analysis
- Signer dependency problem and split design
- MediaPipe keypoint extraction pipeline
- TCN vs BiLSTM comparison
- Training configuration and callbacks
- Full results with graphs and analysis
- Class filtering methodology and overfitting justification

---

## Citation

If you use this work, please cite the original dataset:

```bibtex
@article{sidig2021karsl,
  title   = {KArSL: Arabic Sign Language Database},
  author  = {Sidig, Ala Addin I. and others},
  journal = {ACM Transactions on Asian and Low-Resource
             Language Information Processing},
  volume  = {20},
  number  = {1},
  pages   = {1--19},
  year    = {2021}
}
```

---

## License

This repository is for academic and research purposes.
The KArSL-502 dataset is subject to its own license —
refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)
for terms of use.
