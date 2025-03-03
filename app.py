import os
import time
import tempfile
import streamlit as st
import yt_dlp
import whisper
import openai
from pydub import AudioSegment

class VideoSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.temp_dir = tempfile.mkdtemp()
        self.whisper_model = None  # Will load on demand to save memory
    
    def download_video(self, url, progress_callback=None):
        """Download video from URL using yt-dlp"""
        output_path = os.path.join(self.temp_dir, "video.mp4")
        
        def progress_hook(d):
            if d['status'] == 'downloading' and progress_callback:
                p = d.get('_percent_str', '0%').replace('%', '')
                try:
                    progress_callback(float(p) / 100)
                except:
                    pass
        
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': output_path,
            'quiet': True,
            'progress_hooks': [progress_hook] if progress_callback else [],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Video')
        
        return output_path, title
    
    def extract_audio(self, video_path):
        """Extract audio from video file"""
        audio_path = os.path.join(self.temp_dir, "audio.mp3")
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path
    
    def load_whisper_model(self):
        """Load Whisper model on demand"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        self.load_whisper_model()
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def summarize_text(self, text, title):
        """Summarize text using OpenAI GPT"""
        chunk_size = 4000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        summaries = []
        for chunk in chunks:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an academic assistant that creates detailed summaries of educational content. Include key concepts, examples, and main points."},
                    {"role": "user", "content": f"Create a detailed academic summary of this lecture transcript, organized with headings and bullet points. Include all key concepts, examples, and important information: {chunk}"}
                ],
                max_tokens=1500
            )
            
            summaries.append(response.choices[0].message.content)
            time.sleep(1)  # Avoid rate limits
        
        # Combine and create final summary
        full_summary = "\n\n".join(summaries)
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an academic assistant that creates detailed study guides."},
                {"role": "user", "content": f"Create a comprehensive study guide based on these summaries of '{title}'. Include main topics, key points, definitions, and organize with clear headings and bullet points: {full_summary}"}
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content

# Streamlit UI
def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI Video Summarizer for Students",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
            font-weight: 600;
        }
        .info-text {
            font-size: 1rem;
            color: #424242;
        }
        .summary-container {
            background-color: #F5F7F9;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<div class="main-header">AI Video Summarizer for Students</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Transform educational videos into comprehensive study notes with AI</div>', unsafe_allow_html=True)
    
    # Sidebar for API key
    with st.sidebar:
        st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to use the service")
        
        st.markdown("### About")
        st.markdown("This tool helps students summarize educational videos into comprehensive study notes.")
        st.markdown("##### Features:")
        st.markdown("- Works with videos up to 2 hours long")
        st.markdown("- Supports multiple video platforms")
        st.markdown("- Creates structured academic summaries")
        
        st.markdown("### How It Works")
        st.markdown("1. Enter a video URL")
        st.markdown("2. Wait for the AI to process the video")
        st.markdown("3. Review your comprehensive summary")
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<div class="sub-header">Input</div>', unsafe_allow_html=True)
        video_url = st.text_input("Enter Video URL", placeholder="https://www.youtube.com/watch?v=example")
        
        if st.button("Summarize Video", disabled=not (api_key and video_url)):
            if not api_key or not video_url:
                st.error("Please provide both an API key and a video URL.")
            else:
                # Initialize summarizer
                summarizer = VideoSummarizer(api_key=api_key)
                
                # Processing steps with progress indicators
                with st.status("Processing video...", expanded=True) as status:
                    # Download video
                    st.write("Downloading video...")
                    progress_bar = st.progress(0)
                    
                    def update_progress(p):
                        progress_bar.progress(p)
                    
                    video_path, title = summarizer.download_video(video_url, update_progress)
                    progress_bar.progress(100)
                    
                    # Extract audio
                    st.write("Extracting audio...")
                    audio_path = summarizer.extract_audio(video_path)
                    
                    # Transcribe audio
                    st.write("Transcribing audio (this may take a while)...")
                    transcript = summarizer.transcribe_audio(audio_path)
                    
                    # Summarize content
                    st.write("Generating summary...")
                    summary = summarizer.summarize_text(transcript, title)
                    
                    status.update(label="Processing complete!", state="complete")
                
                # Store the result in session state for display
                st.session_state.summary = summary
                st.session_state.title = title
    
    with col2:
        st.markdown('<div class="sub-header">Summary Output</div>', unsafe_allow_html=True)
        
        # Display summary if available
        if 'summary' in st.session_state:
            st.markdown(f"### {st.session_state.title}")
            st.markdown('<div class="summary-container">', unsafe_allow_html=True)
            st.markdown(st.session_state.summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download option for the summary
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name=f"{st.session_state.title.replace(' ', '_')[:50]}_summary.txt",
                mime="text/plain"
            )
        else:
            st.info("Enter a video URL and click 'Summarize Video' to generate a summary.")
            
            # Show example summary format
            with st.expander("See example summary"):
                st.markdown("""
                # Example Video Title
                
                ## Main Concepts
                * Key concept 1: explanation and details
                * Key concept 2: explanation and details
                
                ## Important Points
                1. First major point discussed in the video
                2. Second major point with examples
                
                ## Summary
                A concise overview of the video content...
                """)

if __name__ == "__main__":
    main()