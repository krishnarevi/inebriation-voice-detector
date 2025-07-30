# Inebriation Voice Detector

## Overview
This project aims to develop a deep learning-based system that classifies a speaker's state of inebriation (drunk or sober) based on a 12-second voice sample. The goal is to build a model capable of detecting a **Blood Alcohol Concentration (BAC)** of ≥ 0.05% with an accuracy of at least **65%**. 

We will leverage a **self-supervised speech model**, fine-tuned specifically on the **Alcohol Language Corpus (ALC)** dataset, to extract useful features and classify the voice samples.

The trained model can be found [here](https://huggingface.co/nagapamel/inebriation-detector) and demo of the application [here](https://nagapamel-drunkenlinguists.hf.space/)
