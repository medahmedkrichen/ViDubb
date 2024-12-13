<div align="center">
    
# ViDubb: Video Dubbing with AI Voice Cloning, Multilingual Features, and Lip-Sync

<p align="center"><img src="Vidubb_img.png" width="1000" height="310">
</div>
<div align="center">
    
|Kaggle|Colab|PRs|
|:-------:|:-------:|:-------:|
|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/medahmedkrichen/vidubb-kaggle-notebook)|[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1--ILLhZuZcruHMH2tpk4_tAD2SHK61EC?authuser=2)|[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
|More memory and GPU time!|Cancel Restarting Session when Asked|Pull Requests Welcome
  
</div>



## Video Dubbing with AI Cloning, Multilingual Capabilities, and Lip-Sync

ViDubb is an advanced AI-powered video dubbing solution focused on delivering high-quality, efficient dubbing for multilingual content. By utilizing cutting-edge voice cloning technology, ViDubb generates realistic voiceovers in multiple languages with exceptional accuracy. The system ensures perfect lip-sync synchronization, matching the voiceovers to the actors' movements, providing a seamless viewing experience. This approach not only enhances the natural flow of dialogue but also preserves the authenticity of the original video. ViDubb streamlines the dubbing process, enabling faster turnaround times while maintaining top-tier audio and visual quality for global audiences.

---

<details open>

<summary>Table of Contents </summary>


- [Certifications](#certifications)
- [The Learning Platform](#the-learning-platform)
- [Reporting Bugs and Issues](#reporting-bugs-and-issues)
- [Reporting Security Issues and Responsible Disclosure](#reporting-security-issues-and-responsible-disclosure)
- [Contributing](#contributing)
- [Platform, Build and Deployment Status](#platform-build-and-deployment-status)
- [License](#license)
  
</details>


## Introduction

**ViDubb** is an advanced AI-powered video dubbing solution designed to deliver high-quality, efficient dubbing for multilingual content. By integrating cutting-edge voice cloning technology and dynamic lip-sync synchronization, ViDubb ensures that voiceovers are perfectly aligned with the original video’s dialogue and actor movements, even when multiple speakers are involved, providing a seamless viewing experience across languages.

Leveraging state-of-the-art AI, **ViDubb** sets new standards in dubbing accuracy and naturalness, making it ideal for global content localization, film, media, and educational purposes. The tool enables content creators and businesses to quickly adapt their videos for international audiences while maintaining top-tier audio and visual quality.

### Key features include:

- **Multi-Language Support**: Offers dubbing in a variety of languages, ensuring broad global accessibility.
- **AI Voice Cloning**: Creates realistic, high-quality voiceovers that capture the tone and emotion of the original content.
- **Dynamic Lip-Sync Technology**: Ensures perfect synchronization with video visuals, even when multiple speakers are involved, enhancing realism and interactivity.
- **Background Sound Preservation**: Retains original background sounds to maintain the authenticity of the video.
- **Efficient Dubbing Process**: Streamlines the video dubbing workflow, enabling faster and more cost-effective localization.
- **Sentence Tokenization**: Breaks down content into manageable segments for better translation and synchronization.
- **Speaker Diarization**: Identifies and separates speakers in the audio, ensuring accurate voice assignment for each speaker during dubbing.
- **Web Interface Support**: Provides an intuitive web interface for easy upload, management, and control of dubbing projects.
- **CPU and GPU Compatibility**: Works seamlessly on both CPU and GPU systems, optimizing performance based on available resources.
- **Download Direct Video from YouTube**: Allows users to download videos directly from YouTube for immediate dubbing and localization, saving time and simplifying the workflow.

Our mission is to provide an efficient and high-quality AI-driven dubbing solution that empowers content creators to expand their global reach, bringing videos to audiences in multiple languages with perfect synchronization and immersive quality.


---

## TO DO LIST

- [ ] Implement sentence summarization.
- [ ] Improve the Dynamic Lip-Sync Technology with a lot of speakers.
- [ ] Deploy ViDubb on HuggingFace space

---

## Examples

| Original Video                                               | ViDubb                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <video src=""> | <video src=""> |

---

## ViDubb Installation and Usage Guide

ViDubb is an AI-powered video dubbing project that involves voice cloning, multilingual capabilities, lip-syncing, and background sound preservation. Follow the steps below to set up and run ViDubb.

### 0) Install Anaconda
Before starting, ensure you have [Anaconda](https://docs.anaconda.com/anaconda/install/) installed on your system. Anaconda is used to manage Python environments and dependencies.

### 1) Set Up the Conda Environment
1. **Remove any existing environment** (if necessary):
    ```bash
    conda remove -n vidubbtest --all
    ```

2. **Create a new conda environment** with Python 3.10.14 and IPython:
    ```bash
    conda create -n "vidubbtest" python=3.10.14 ipython
    ```

3. **Activate the environment**:
    ```bash
    conda activate vidubbtest
    ```

### 2) Clone the Repository
1. **Clone the ViDubb repository** from GitHub:
    ```bash
    git clone https://github.com/medahmedkrichen/ViDubb.git
    ```

2. **Navigate to the ViDubb directory**:
    ```bash
    cd ViDubb
    ```

### 3) Configure the `.env` File
1. **Set up the `.env` file** with your Hugging Face API and Groq API tokens:
    - Create a `.env` file in the `ViDubb` directory.
    - Add the following lines:
    ```bash
    HF_TOKEN="your_huggingface_token"
    Groq_TOKEN="your_groq_token"
    ```

### 4) Install Dependencies
1. **Install FFmpeg** (for audio/video processing):
    ```bash
    sudo apt-get install ffmpeg
    ```

2. **Install Python dependencies** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### 5) Configure CUDA for GPU Acceleration
1. **Install PyTorch with CUDA support** for GPU acceleration:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2. **Check if CUDA is available**:
    Open a Python shell and run the following:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

### 6) Download Wave2Lip Models
1. **Download the Wav2Lip model**:
    ```bash
    wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O 'Wav2Lip/wav2lip_gan.pth'
    ```

2. **Download the face detection model**:
    ```bash
    wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
    ```

### 7) Run the Project
1. **Run the inference script** to process a video:
    ```bash
    python inference.py --yt_url "https://www.youtube.com/shorts/MCS5K9nWfGM" --source_language "en" --target_language "fr" --LipSync True
    ```
    This command will:
    - Download the video from YouTube.
    - Perform lip-sync translation from English to French.
    - Output a dubbed video with lip-syncing.

### 8) Launch the Gradio Web App
1. **Start the web application**:
    ```bash
    python app.py
    ```

2. **Access the app** by opening a browser and going to:
    ```
    http://localhost:7860/
    ```

---

By following these steps, you should be able to set up and run ViDubb for video dubbing with AI-powered voice and lip synchronization.



---

### License

Copyright © 2024 
