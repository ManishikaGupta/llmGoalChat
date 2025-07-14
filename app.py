import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import whisper
import numpy as np
import av
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Whisper model (use 'tiny' or 'base' for faster response)
asr_model = whisper.load_model("base")

# Set up page
st.set_page_config(page_title="Voice Budget Assistant", layout="centered")
st.title("🎙️ Voice Budget Assistant")
st.markdown("Click **Start**, speak your income and financial goals, then click **Transcribe and Ask Gemini**.")

# Domain prompt
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

# Audio processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        self.frames.append(frame.to_ndarray().flatten())

    def transcribe(self):
        if not self.frames:
            return ""
        audio = np.concatenate(self.frames).astype(np.float32) / 32768.0
        result = asr_model.transcribe(audio)
        return result["text"]

# Store processor in session
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

# Streamlit WebRTC
webrtc_ctx = webrtc_streamer(
    key="mic-input",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    audio_frame_callback=st.session_state.audio_processor.recv,
    audio_receiver_size=1024,
    sendback_audio=False,
)

# Transcribe and respond
if st.button("📝 Transcribe and Ask Gemini"):
    with st.spinner("Transcribing your speech..."):
        user_input = st.session_state.audio_processor.transcribe()

    if user_input.strip():
        st.success(f"🗣️ You said: {user_input}")
        st.chat_message("user").markdown(user_input)

        prompt = f"{DOMAIN_PROMPT}\n\nUser: {user_input}"

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"⚠️ Gemini Error: {e}"

        st.chat_message("assistant").markdown(bot_reply)
    else:
        st.warning("Couldn't capture your speech. Please try again.")
