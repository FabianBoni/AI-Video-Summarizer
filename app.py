import os
import time
import tempfile
import streamlit as st
import yt_dlp
import whisper
from openai import OpenAI
from pydub import AudioSegment
from fpdf import FPDF

class VideoSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.temp_dir = tempfile.mkdtemp()
        self.whisper_model = None  # Will load on demand to save memory
        self.client = OpenAI(api_key=api_key)

    def create_pdf(self, summary, title):
        pdf = FPDF()
        pdf.add_page()
        
        # Clean text function to handle Unicode characters
        def clean_text(text):
            # Replace problematic characters with their closest ASCII equivalents
            text = text.replace('\u201e', '"')  # German opening quote
            text = text.replace('\u201c', '"')  # Opening double quote
            text = text.replace('\u201d', '"')  # Closing double quote
            text = text.replace('\u2018', "'")  # Opening single quote
            text = text.replace('\u2019', "'")  # Closing single quote
            text = text.replace('\u2013', '-')  # En dash
            text = text.replace('\u2014', '--')  # Em dash
            text = text.replace('\u2026', '...')  # Ellipsis
            text = text.replace('\u00a0', ' ')  # Non-breaking space
            
            # Filter out any remaining characters not in latin-1
            return ''.join(c if ord(c) < 256 else '?' for c in text)
        
        # Clean the title and summary
        clean_title = clean_text(title)
        clean_summary = clean_text(summary)
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, txt=clean_title, ln=True, align="C")
        pdf.ln(10)
        
        # Add summary text
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, clean_summary)
        
        # Return the PDF as bytes with error handling for encoding
        pdf_bytes = pdf.output(dest="S").encode("latin1", errors="replace")
        return pdf_bytes

    def download_video(self, url, progress_callback=None):
        """Download video from URL using yt-dlp with enhanced error handling"""
        output_path = os.path.join(self.temp_dir, "video.mp4")
        
        def progress_hook(d):
            if d['status'] == 'downloading' and progress_callback:
                p = d.get('_percent_str', '0%').replace('%', '')
                try:
                    progress_callback(float(p) / 100)
                except:
                    pass
        
        # Enhanced options for better compatibility
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # More flexible format selection
            'outtmpl': output_path,
            'quiet': False,  # Set to False to see detailed output for debugging
            'progress_hooks': [progress_hook] if progress_callback else [],
            'no_warnings': False,
            'ignoreerrors': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to extract info without downloading to verify the URL works
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise Exception("Failed to extract video information")
                
                # If verification succeeded, proceed with download
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Video')
                
                # Verify the file exists and has content
                if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                    raise Exception("Downloaded file is missing or too small")
                
                return output_path, title
        except Exception as e:
            # Provide detailed error information
            st.error(f"Video download failed: {str(e)}")
            raise Exception(f"Failed to download video: {str(e)}")

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        audio_path = os.path.join(self.temp_dir, "audio.mp3")
        video = AudioSegment.from_file(video_path)
        video.export(audio_path, format="mp3")
        return audio_path

    def load_whisper_model(self):
        """Load Whisper model on demand with GPU acceleration"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base", device="cuda")

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper with GPU acceleration"""
        self.load_whisper_model()
        result = self.whisper_model.transcribe(audio_path, fp16=True)
        return result["text"]
    
    def summarize_text(self, text, title, format_type="bullet_points", language="en"):
        """Summarize text using OpenAI GPT with multiple calls: first to break down the transcript and then to produce detailed study notes."""
        import concurrent.futures
        import queue

        # Create a placeholder for live updates
        live_output_placeholder = st.empty()

        chunk_size = 4000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Set language-specific prompts for the first summarization step
        if language == "en":
            system_prompt = ("You are an academic assistant that summarizes lecture transcripts. "
                            "Highlight both what was shown and add interpretations that help students learn.")
            format_instruction = "bullet points" if format_type == "bullet_points" else "continuous paragraphs"
            user_prompt = (f"Create a detailed academic summary of this lecture transcript. "
                        f"Focus on the key points and explain the concepts in a way that helps students create study notes. "
                        f"Organize the summary with {format_instruction}: ")
            # Second, follow-up prompt to further develop detailed study notes for learning
            followup_system_prompt = ("You are an academic mentor. Based on the summary provided, "
                                    "elaborate detailed study notes detailing why each point is important for learning, "
                                    "including examples, definitions, and methods to remember the content.")
            followup_user_prompt = f"Based on the following summary of '{title}', create comprehensive, detailed study notes:\n"
        else:  # German version
            system_prompt = ("Du bist ein akademischer Assistent, der Vortragsmanuskripte zusammenfasst. "
                            "Hebe sowohl die gezeigten Inhalte hervor als auch Interpretationen, die dem Lernen helfen.")
            format_instruction = "Aufzählungspunkte" if format_type == "bullet_points" else "fortlaufende Absätze"
            user_prompt = (f"Erstelle eine detaillierte akademische Zusammenfassung dieses Vortragsmanuskripts. "
                        f"Fokussiere auf die wichtigsten Punkte und erläutere die Konzepte so, dass sie als Lernnotizen dienen können. "
                        f"Organisiere die Zusammenfassung mit {format_instruction}: ")
            followup_system_prompt = ("Du bist ein akademischer Mentor. Basierend auf der Zusammenfassung, "
                                    "erarbeite detaillierte Lernnotizen, in denen du erklärst, warum die einzelnen Punkte wichtig sind, "
                                    "mit Beispielen, Definitionen und Methoden zum besseren Einprägen des Inhalts.")
            followup_user_prompt = f"Erstelle auf Grundlage der folgenden Zusammenfassung von '{title}' umfassende Lernnotizen:\n"

        # Thread-safe communication channel for first stage
        result_queue = queue.Queue()
        results = [None] * len(chunks)

        live_output_placeholder.markdown(f"""
        ## Starting: {title}
        Processing chunks: 0/{len(chunks)} completed
        
        <div class="summary-container">
        Generating initial summary...
        </div>
        """, unsafe_allow_html=True)

        # Function to process each chunk (runs in worker threads)
        def process_chunk(chunk, index):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + chunk}
                        ],
                        max_tokens=1500
                    )
                    result_queue.put((index, response.choices[0].message.content))
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        result_queue.put((index, f"Error processing chunk {index+1}: {str(e)}"))
                        return
                    time.sleep(2)

        # Launch processing threads for the first stage
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_chunk, chunk, i) for i, chunk in enumerate(chunks)]
            
            completed = 0
            while completed < len(chunks):
                try:
                    index, content = result_queue.get(timeout=0.5)
                    results[index] = content
                    completed += 1
                    display_parts = []
                    for i, result in enumerate(results):
                        if result:
                            display_parts.append(f"### Part {i+1}:\n{result}")
                    
                    live_output_placeholder.empty()
                    live_output_placeholder.markdown(f"""
                    ## In Progress: {title}
                    Processing chunks: {completed}/{len(chunks)} completed
                    
                    <div class="summary-container">
                    {"<hr>".join(display_parts)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except queue.Empty:
                    continue
            concurrent.futures.wait(futures)
        
        valid_results = [r for r in results if r and not r.startswith("Error")]
        initial_summary = "\n\n".join(valid_results)
        
        live_output_placeholder.markdown(f"""
        ## Finalizing Initial Summary: {title}
        Preparing detailed study notes...
        
        <div class="summary-container">
        {initial_summary}
        </div>
        """, unsafe_allow_html=True)
        
        # Second phase: Generate detailed study notes using a follow-up OpenAI call
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": followup_system_prompt},
                {"role": "user", "content": followup_user_prompt + initial_summary}
            ],
            max_tokens=2000
        )
        final_summary = response.choices[0].message.content
        
        live_output_placeholder.markdown(f"""
        ## Complete: {title}
        
        <div class="summary-container">
        {final_summary}
        </div>
        """, unsafe_allow_html=True)
        
        return final_summary

# Streamlit UI
def main():
    # Define language_code with a default value at the beginning of the function
    language_code = "en"  # Default to English

    # Now you can safely use language_code for page configuration
    page_title = "AI Video Summarizer for Students" if language_code == "en" else "KI-Video-Zusammenfasser für Studierende"
    page_icon = "🎓"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
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

    # App header in the selected language
    header_text = "AI Video Summarizer for Students" if language_code == "en" else "KI-Video-Zusammenfasser für Studierende"
    info_text = "Transform educational videos into comprehensive study notes with AI" if language_code == "en" else "Verwandle Lehrvideos in umfassende Lernnotizen mit KI"

    st.markdown(f'<div class="main-header">{header_text}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">{info_text}</div>', unsafe_allow_html=True)

    # Sidebar for API key
    with st.sidebar:
        st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to use the service")

        # Language selection - this will update the default value we set earlier
        st.markdown("### Language / Sprache")
        language = st.radio(
            "Select language / Sprache auswählen",
            options=["English", "Deutsch"],
            index=0  # Default to English
        )
        language_code = "en" if language == "English" else "de"

        # Format selection
        format_label = "Summary Format" if language == "English" else "Zusammenfassungsformat"
        format_type = st.radio(
            format_label,
            options=["Bullet Points", "Continuous Text"] if language == "English" else ["Aufzählungspunkte", "Fließender Text"],
            index=0  # Default to bullet points
        )
        format_code = "bullet_points" if format_type in ["Bullet Points", "Aufzählungspunkte"] else "continuous_text"

        # About section in the selected language
        if language_code == "en":
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
        else:
            st.markdown("### Über")
            st.markdown("Dieses Tool hilft Studierenden, Lehrvideos in umfassende Lernnotizen zusammenzufassen.")
            st.markdown("##### Funktionen:")
            st.markdown("- Funktioniert mit Videos bis zu 2 Stunden Länge")
            st.markdown("- Unterstützt mehrere Videoplattformen")
            st.markdown("- Erstellt strukturierte akademische Zusammenfassungen")

            st.markdown("### Wie es funktioniert")
            st.markdown("1. Geben Sie eine Video-URL ein")
            st.markdown("2. Warten Sie, bis die KI das Video verarbeitet hat")
            st.markdown("3. Überprüfen Sie Ihre umfassende Zusammenfassung")    
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
                    download_progress = st.progress(0)

                    def update_download_progress(p):
                        download_progress.progress(p)

                    video_path, title = summarizer.download_video(video_url, update_download_progress)
                    download_progress.progress(100)

                    # Extract audio
                    st.write("Extracting audio...")
                    extract_progress = st.progress(0)
                    for i in range(100):
                        # Simulate progress during audio extraction
                        time.sleep(0.01)
                        extract_progress.progress(i + 1)
                    audio_path = summarizer.extract_audio(video_path)

                    # Transcribe audio
                    st.write("Transcribing audio (this may take a while)...")
                    transcribe_progress = st.progress(0)
                    # Add a placeholder for estimated time
                    time_placeholder = st.empty()

                    start_time = time.time()
                    transcript = summarizer.transcribe_audio(audio_path)

                    # Update progress periodically during transcription
                    for i in range(100):
                        elapsed = time.time() - start_time
                        estimated = (elapsed / (i+1)) * 100 if i > 0 else 0
                        time_placeholder.text(f"Estimated time remaining: {estimated:.1f} seconds")
                        time.sleep(elapsed/100)  # Dynamic sleep based on processing time
                        transcribe_progress.progress(i + 1)

                    # Summarize content
                st.write("Generating summary...")
                summary_progress = st.progress(0)
                # Simulate progress while the summary is being generated.
                for i in range(100):
                    time.sleep(0.02)
                    summary_progress.progress(i + 1)
                # Generate the summary only once after progress completes.
                summary = summarizer.summarize_text(
                        transcript, 
                        title, 
                        format_type=format_code,
                        language=language_code
                )
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

            # Download button for text version
            st.download_button(
                label="Download Summary as Text",
                data=st.session_state.summary,
                file_name=f"{st.session_state.title.replace(' ', '_')[:50]}_summary.txt",
                mime="text/plain"
            )
        
            # Create a new summarizer instance specifically for PDF creation
            pdf_summarizer = VideoSummarizer(api_key=api_key)
            pdf_bytes = pdf_summarizer.create_pdf(st.session_state.summary, st.session_state.title)
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_bytes,
                file_name=f"{st.session_state.title.replace(' ', '_')[:50]}_summary.pdf",
                mime="application/pdf"
            )
        else:            st.info("Enter a video URL and click 'Summarize Video' to generate a summary.")

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