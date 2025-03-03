# AI Video Summarizer

An intelligent application that transforms educational videos into comprehensive study notes with AI assistance.

## Created by Fabian Boni

This project was developed by Fabian Boni to help students efficiently process educational content from online videos.

## Features
- Summarizes videos from any online source up to 2 hours in length
- Creates structured academic summaries with key points and concepts
- Beautiful user interface for easy interaction
- Real-time progress tracking
- Downloadable summary text

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio processing)
- OpenAI API key
- CUDA (required for GPU acceleration)

### Installing FFmpeg on Windows with Chocolatey
The easiest way to install FFmpeg on Windows is using Chocolatey:

First, install Chocolatey package manager by running this in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
```

Install FFmpeg with a single command:
```powershell
choco install ffmpeg
```

Verify the installation by opening a new command prompt and typing:
```powershell
ffmpeg -version
```

### Installing CUDA
To enable GPU acceleration, install NVIDIA CUDA:

1. Download the latest CUDA Toolkit from the official NVIDIA website:  
   [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Follow the installation instructions for your operating system.
3. Verify the installation by running:
   ```bash
   nvcc --version
   ```

### Setting up the Application
Clone the repository:
```bash
git clone https://github.com/fabianboni/ai-video-summarizer.git
cd ai-video-summarizer
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
Launch the application:
```bash
streamlit run app.py
```

1. Enter your OpenAI API key in the sidebar
2. Paste the URL of the educational video you want to summarize
3. Click "Summarize Video" and wait for processing to complete
4. View your structured summary in the output window
5. Download the summary as a text file if needed

## Credits
This project was developed by Fabian Boni. All rights reserved.

**Concept & Implementation:** Fabian Boni  
**Video Processing:** FFmpeg & yt-dlp  
**Transcription:** OpenAI Whisper  
**Summarization:** OpenAI GPT  
**User Interface:** Streamlit  

For questions or feedback, please contact Fabian Boni.

## License
Copyright © 2023 Fabian Boni. All rights reserved.