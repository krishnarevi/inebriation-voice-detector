# 🍷 Detecting Alcohol Inebriation from Speech

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Wav2Vec2-green.svg)](https://huggingface.co/facebook/wav2vec2-base)
[![Dataset](https://img.shields.io/badge/Dataset-ALC-orange.svg)](https://www.phonetik.uni-muenchen.de/Bas/BasALCeng.html)

## 🚀 Overview

Alcohol impairment is a critical factor in road accidents, and existing detection systems are often intrusive or impractical for real-time use. This project explores **speech-based alcohol detection** as a non-invasive alternative.

By fine-tuning a **Wav2Vec2** model, we trained a binary classifier to distinguish between **inebriated** and **sober** speech (BAC ≥ 0.05%). Our approach achieves **state-of-the-art results** on the Alcohol Language Corpus (ALC), with strong generalization across demographics and speech types.

---

## 📚 Dataset

We use the **Alcohol Language Corpus (ALC)**, a controlled dataset of German speech recordings from speakers in both sober and intoxicated states. It includes:

- Read and spontaneous speech
- Multiple speakers
- Demographic variation
- BAC measurements for each session

> 🔗 [Dataset Info](https://www.phonetik.uni-muenchen.de/Bas/BasALCeng.html)

---

## 🧠 Model

We fine-tuned the **Wav2Vec2** base model for binary classification:

- **Architecture**: Wav2Vec2 + Classification Head
- **Loss**: Cross-Entropy
- **Optimization Target**: Unweighted Average Recall (UAR)

### 🔧 Features

- ✅ Fine-tuned **self-supervised speech representations**
- ✅ **High accuracy** on imbalanced data
- ✅ Robust across **demographics & speech types**
- ✅ Lightweight & inference-ready

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **UAR (Test)** | **75.7%** |
| **Baseline UAR** | ~67% |
| **Improvement** | +8.7% |

> 📈 This is the **highest reported performance** on the ALC dataset for binary alcohol detection.

---


### 🌐 Live Demo & Resources
Hugging Face Demo: Try out the live demo[here](https://huggingface.co/nagapamel/inebriation-detector)

