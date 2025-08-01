# Detecting Alcohol Inebriation from Speech
### 🚀 Project Overview
Driving under the influence of alcohol presents a significant public safety hazard. Traditional detection methods are often intrusive and impractical for continuous monitoring. Speech, however, offers a non-invasive alternative as alcohol consumption significantly alters prosody, fluency, and articulation.

In this work, we developed a binary classifier to identify inebriated speech (defined as speech produced by speakers with a BAC of ≥ 0.05%). By fine-tuning a Wav2Vec2 model, we optimized for Unweighted Average Recall (UAR) and achieved a UAR of 75.69% on the ALC dataset, setting a new state-of-the-art for this task. The model also demonstrates consistent performance across various demographics and speech types.

### 📚 Dataset
The project utilizes the Alcohol Language Corpus (ALC), a collection of German speech recordings from both sober and intoxicated speakers. 

### ✨ Features

**Wav2Vec2 Fine-tuning:** Leverages powerful pre-trained speech representations for superior accuracy.

**Robust & Generalizable:** Demonstrated consistent performance across different demographic groups and speech types.

**State-of-the-Art Performance:** Achieves 75.7% Unweighted Average Recall (UAR) on the ALC dataset, setting a new benchmark.

### 📊 Results
Our model significantly outperforms previous methods on the Alcohol Language Corpus (ALC). We've achieved a UAR of 75.7%, which is currently the highest reported performance on this dataset for binary inebriation detection. 

### 🌐 Live Demo & Resources
Hugging Face Demo: Try out the live demo[here](https://huggingface.co/nagapamel/inebriation-detector)

