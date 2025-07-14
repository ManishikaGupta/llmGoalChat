import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import numpy as np
import av
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Whisper ASR model
asr_model = whisper.load_model("base")

# Streamlit page setup
st.set_page_config(page_title="🎧 Voice Budget Assistant", layout="centered")
st.title("🎙️ Speak Your Budget Goals")
st.markdown("Click **Start Recording**, say your income and financial goals, then click **Transcribe and Ask Gemini**.")

# Domain prompt for Gemini
DOMAIN_PROMPT = """
You are a helpful and knowledgeable financial advisor.

Your task is to assist users in optimizing their monthly budgets based on their income and specific financial goals. When the user provides this information, respond with a clear, actionable, and realistic monthly budget plan.

Break the user's income into the following categories:
- Essentials (e.g. rent, groceries, bills)
- Debt Repayment (e.g. loans, EMIs)
- Savings (e.g. emergency fund, short-term needs)
- Investments (e.g. SIPs, mutual funds)
- Lifestyle (e.g. dining, shopping, travel)

Explain how each part supports the user's goals. Make sure your recommendations are practical and aligned with their income level and timelines.
"""

# Audio buffer class
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        self.frames.append(frame.to_ndarray().flatten())

    def get_transcription(self):
        if not self.frames:
            return ""
        audio = np.concatenate(self.frames).astype(np.float32) / 32768.0
        result = asr_model.transcribe(audio)
        return result["text"]

# Initialize audio processor and session state
processor = AudioProcessor()
if "recording" not in st.session_state:
    st.session_state.recording = False

# Start/stop buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎤 Start Recording"):
        st.session_state.recording = True
with col2:
    if st.button("🛑 Stop Recording"):
        st.session_state.recording = False

# Show current status
if st.session_state.recording:
    st.success("🎙️ Recording... Speak now.")
else:
    st.info("🛑 Not Recording.")

# Start webrtc_streamer only if recording
if st.session_state.recording:
    webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        audio_receiver_size=256,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        audio_frame_callback=processor.recv,
        sendback_audio=False
    )

# Transcribe and respond with Gemini
if st.button("📝 Transcribe and Ask Gemini"):
    text = processor.get_transcription()
    if text.strip():
        st.success(f"🗣️ You said: {text}")
        st.chat_message("user").markdown(text)

        full_prompt = f"{DOMAIN_PROMPT}\n\nUser: {text}"
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(full_prompt)
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"⚠️ Gemini Error: {e}"

        st.chat_message("assistant").markdown(bot_reply)
    else:
        st.warning("Couldn't detect any speech. Please try again.")
