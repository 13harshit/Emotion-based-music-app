import streamlit as st
import sys
import os
import webbrowser
from datetime import datetime, timedelta
import numpy as np
import cv2
import av
import mediapipe as mp
from keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import auth functions
from images.auth import is_authenticated, show_auth_page, logout, get_database

# Optional speech recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except Exception:
    SPEECH_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="MoodBeats - AI Music Player", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS styling inspired by latest design trends
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-light: rgba(255, 255, 255, 0.9);
        --text-muted: rgba(255, 255, 255, 0.7);
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 0 !important;
        background: var(--primary);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: none !important;
    }
    
    /* Hide Streamlit branding */
    header[data-testid="stHeader"],
    .stApp > header,
    footer,
    .stDeployButton {
        display: none !important;
        height: 0 !important;
    }
    
    /* Modern Navigation Header */
    .navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        background: rgba(255, 255, 255, 0.1);
        border-bottom: 1px solid var(--glass-border);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .logo {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #fff, #a8edea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(255,255,255,0.6)); }
    }
    
    .nav-menu {
        display: flex;
        gap: 0.5rem;
        background: var(--glass);
        padding: 0.5rem;
        border-radius: 50px;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(10px);
    }
    
    .nav-item {
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        color: var(--text-muted);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background: transparent;
        position: relative;
        overflow: hidden;
    }
    
    .nav-item:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .nav-item:hover:before {
        left: 100%;
    }
    
    .nav-item.active, .nav-item:hover {
        background: var(--glass-border);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .user-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: var(--glass);
        padding: 0.75rem 1.25rem;
        border-radius: 50px;
        border: 1px solid var(--glass-border);
        color: white;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Main Container */
    .app-wrapper {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
        min-height: calc(100vh - 100px);
    }
    
    /* Hero Section */
    .hero {
        text-align: center;
        padding: 4rem 2rem;
        margin-bottom: 3rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(30px);
        border-radius: 30px;
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .hero:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: rotate 10s linear infinite;
        pointer-events: none;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
    }
    
    .hero-title {
        font-size: clamp(2rem, 5vw, 4rem);
        font-weight: 800;
        background: linear-gradient(45deg, #fff, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-muted);
        margin-bottom: 2rem;
        font-weight: 300;
        line-height: 1.6;
    }
    
    /* Glass Cards */
    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fadeInScale 0.6s ease-out;
    }
    
    @keyframes fadeInScale {
        from { transform: scale(0.95); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .glass-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    }
    
    /* Enhanced Emotion Badge */
    .emotion-display {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 1rem;
        background: var(--secondary);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.4);
        animation: pulse 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .emotion-badge:before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #f093fb, #f5576c, #4facfe, #00f2fe);
        border-radius: 50px;
        z-index: -1;
        animation: rotate 3s linear infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .emotion-icon {
        font-size: 1.5rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    /* Modern Form Elements */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background: var(--glass) !important;
        border: 2px solid var(--glass-border) !important;
        border-radius: 16px !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: rgba(79, 172, 254, 0.8) !important;
        box-shadow: 0 0 0 4px rgba(79, 172, 254, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
        font-weight: 400 !important;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 48px rgba(79, 172, 254, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98) !important;
    }
    
    /* Platform Grid */
    .platform-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .platform-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .platform-card:hover {
        transform: translateY(-8px) rotateX(5deg);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        border-color: var(--glass-border);
    }
    
    .platform-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        transition: all 0.3s ease;
    }
    
    .platform-card:hover .platform-icon {
        transform: scale(1.2) rotateY(360deg);
    }
    
    .platform-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .platform-description {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Games Grid */
    .games-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .game-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .game-card:hover {
        transform: translateY(-12px) rotateX(5deg);
        box-shadow: 0 25px 80px rgba(0, 0, 0, 0.3);
    }
    
    .game-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        transition: all 0.4s ease;
    }
    
    .game-card:hover .game-image {
        transform: scale(1.1);
    }
    
    .game-content {
        padding: 2rem;
        position: relative;
    }
    
    .game-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }
    
    .game-description {
        color: var(--text-muted);
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .play-button {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--secondary);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
    }
    
    .play-button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4);
    }
    
    /* Statistics Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.2);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(30px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.1);
    }
    
    /* Profile Section */
    .profile-header {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 3rem;
        padding: 2rem;
        background: var(--glass);
        border-radius: 20px;
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
    }
    
    .profile-avatar-large {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: var(--secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
    }
    
    .profile-info h2 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .profile-info p {
        color: var(--text-muted);
        font-size: 1.1rem;
    }
    
    /* Video Container */
    .video-container {
        background: var(--glass);
        backdrop-filter: blur(20px);
        border: 2px solid var(--glass-border);
        border-radius: 24px;
        padding: 1.5rem;
        margin: 2rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .video-container:before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #4facfe, #00f2fe, #f093fb, #f5576c);
        border-radius: 24px;
        z-index: -1;
        animation: rotate 5s linear infinite;
    }
    
    /* Labels */
    .stSelectbox > label, 
    .stTextInput > label,
    .stTextArea > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Alerts */
    .stSuccess, .stError, .stWarning, .stInfo {
        backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        border: 1px solid var(--glass-border) !important;
        font-weight: 500 !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        background: var(--glass) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        border: 1px solid var(--glass-border) !important;
        overflow: hidden !important;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .app-wrapper {
            padding: 1rem;
        }
        
        .navbar {
            padding: 1rem;
            flex-direction: column;
            gap: 1rem;
        }
        
        .nav-menu {
            order: -1;
            width: 100%;
            justify-content: center;
        }
        
        .hero {
            padding: 2rem 1rem;
        }
        
        .hero-title {
            font-size: 2rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .profile-header {
            flex-direction: column;
            text-align: center;
        }
        
        .games-grid {
            grid-template-columns: 1fr;
        }
        
        .platform-grid {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4facfe;
    }
    
    /* Custom animations for page transitions */
    .page-transition {
        animation: pageSlide 0.5s ease-out;
    }
    
    @keyframes pageSlide {
        from {
            transform: translateX(20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Helper Functions ------------------
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("model.h5")
        labels = np.load("labels.npy")
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, labels = load_emotion_model()

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Database collections
def get_history_collection():
    db = get_database()
    return db['emotion_history'] if db else None

def get_user_preferences_collection():
    db = get_database()
    return db['user_preferences'] if db else None

# Emotion processor for WebRTC
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            if model is not None:
                pred = labels[np.argmax(model.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                np.save("emotion.npy", np.array([pred]))
                
                # Save to database
                try:
                    col = get_history_collection()
                    if col is not None:
                        username = st.session_state.get('username', 'anonymous')
                        entry = {
                            'username': username,
                            'emotion': str(pred),
                            'timestamp': datetime.utcnow(),
                            'language': st.session_state.get('pref_lang', ''),
                            'singer': st.session_state.get('pref_singer', ''),
                        }
                        col.insert_one(entry)
                except Exception:
                    pass

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                              landmark_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=1),
                              connection_drawing_spec=drawing.DrawingSpec(thickness=1, color=(255, 255, 255)))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Voice to text function
def speech_to_text():
    if not SPEECH_AVAILABLE:
        return None, "Speech recognition not available"
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        
        text = r.recognize_google(audio)
        return text, "Success"
    except sr.WaitTimeoutError:
        return None, "No speech detected"
    except sr.UnknownValueError:
        return None, "Could not understand speech"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ------------------ Authentication Check ------------------
if not is_authenticated():
    show_auth_page()
    st.stop()

username = st.session_state.get('username', 'Unknown User')

# Initialize navigation state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# ------------------ Navigation Header ------------------
st.markdown(f"""
<div class="navbar">
    <div class="logo">🎵 MoodBeats</div>
    <div class="nav-menu">
        <button class="nav-item {'active' if st.session_state.current_page == 'Home' else ''}" onclick="setPage('Home')">🏠 Home</button>
        <button class="nav-item {'active' if st.session_state.current_page == 'Games' else ''}" onclick="setPage('Games')">🎮 Games</button>
        <button class="nav-item {'active' if st.session_state.current_page == 'History' else ''}" onclick="setPage('History')">📜 History</button>
        <button class="nav-item {'active' if st.session_state.current_page == 'Analytics' else ''}" onclick="setPage('Analytics')">📊 Analytics</button>
        <button class="nav-item {'active' if st.session_state.current_page == 'Profile' else ''}" onclick="setPage('Profile')">👤 Profile</button>
    </div>
    <div class="user-badge">
        <div class="user-avatar">{username[0].upper()}</div>
        <span>{username}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons (fallback for when JS doesn't work)
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([1, 1, 1, 1, 1, 2])
with nav_col1:
    if st.button("🏠 Home", key="nav_home"):
        st.session_state.current_page = 'Home'
with nav_col2:
    if st.button("🎮 Games", key="nav_games"):
        st.session_state.current_page = 'Games'
with nav_col3:
    if st.button("📜 History", key="nav_history"):
        st.session_state.current_page = 'History'
with nav_col4:
    if st.button("📊 Analytics", key="nav_analytics"):
        st.session_state.current_page = 'Analytics'
with nav_col5:
    if st.button("👤 Profile", key="nav_profile"):
        st.session_state.current_page = 'Profile'
with nav_col6:
    if st.button("🚪 Logout", key="nav_logout"):
        logout()

# ------------------ Game Data ------------------
GAMES = [
    {
        "name": "Geometry Dash",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/d3b19eca4e4e15cdde3f3dc245b3b0f1.png",
        "url": "https://poki.com/en/g/geometry-dash-lite",
        "description": "Jump and fly through danger in this rhythm-based action platformer! Navigate through challenging levels with perfect timing."
    },
    {
        "name": "Subway Surfers",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/4a2a60b7-6e39-4d00-836a-36a6bb715ad1.png",
        "url": "https://poki.com/en/g/subway-surfers",
        "description": "Run as far as you can in this endless runner adventure! Dodge trains and collect coins in this exciting chase."
    },
    {
        "name": "Temple Run 2",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/b9dc9f05d6c2d6b3e4c8b8e7a7c87ae5.png",
        "url": "https://poki.com/en/g/temple-run-2",
        "description": "Navigate perilous cliffs, zip lines, mines and forests! The ultimate endless running adventure continues."
    },
    {
        "name": "Cut the Rope",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/f40d966ecbb74fe7b0e90c1d8b87d6f5.png",
        "url": "https://poki.com/en/g/cut-the-rope",
        "description": "Feed candy to Om Nom in this award-winning puzzle game! Use physics and timing to solve each level."
    },
    {
        "name": "Moto X3M",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/5eb54a8ec0d2c3c06ae24e8e92b25c1e.png",
        "url": "https://poki.com/en/g/moto-x3m",
        "description": "Bike racing with amazing stunts and obstacles! Perform incredible tricks while racing through challenging courses."
    },
    {
        "name": "Fireboy and Watergirl",
        "image": "https://img.poki.com/cdn-cgi/image/quality=78,width=314,height=314,fit=cover,f=auto/85de20dc5ae02a2a9cb9cb47b4e5e4cc.png",
        "url": "https://poki.com/en/g/fireboy-and-water-girl-the-forest-temple",
        "description": "Teamwork puzzle adventure in the Forest Temple! Control both characters to solve challenging puzzles together."
    }
]

# ------------------ Main App Container ------------------
st.markdown('<div class="app-wrapper page-transition">', unsafe_allow_html=True)

# ------------------ HOME PAGE ------------------
if st.session_state.current_page == 'Home':
    # Hero section
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">AI-Powered Music Discovery</h1>
            <p class="hero-subtitle">Let advanced AI analyze your emotions through facial expressions and find the perfect music to match your current mood</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current emotion status
    try:
        current_emotion = np.load("emotion.npy")[0]
    except:
        current_emotion = ""
    
    if current_emotion:
        st.markdown(f"""
        <div class="emotion-display">
            <div class="emotion-badge">
                <span class="emotion-icon">🎭</span>
                <span>Current Mood: {current_emotion.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎼 Music Preferences")
        
        # Voice input section
        if SPEECH_AVAILABLE:
            voice_col1, voice_col2 = st.columns([3, 1])
            with voice_col2:
                if st.button("🎤 Voice Input", key="voice_btn"):
                    with st.spinner("Listening..."):
                        text, status = speech_to_text()
                    if text:
                        st.success(f"Heard: {text}")
                        # Simple parsing
                        if " by " in text:
                            parts = text.split(" by ")
                            st.session_state['pref_lang'] = parts[0].replace("song", "").strip()
                            st.session_state['pref_singer'] = parts[1].strip()
                        else:
                            st.session_state['pref_singer'] = text
                    else:
                        st.error(status)
        
        # Text inputs
        lang = st.text_input(
            "🌍 Preferred Language", 
            placeholder="e.g., Hindi, English, Spanish, Korean",
            value=st.session_state.get('pref_lang', ''),
            key="lang_input"
        )
        singer = st.text_input(
            "🎤 Favorite Singer/Artist", 
            placeholder="e.g., Arijit Singh, Taylor Swift, BTS, Adele",
            value=st.session_state.get('pref_singer', ''),
            key="singer_input"
        )
        
        st.session_state['pref_lang'] = lang
        st.session_state['pref_singer'] = singer
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Camera section - Show only if preferences are set
        if lang and singer:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📹 Emotion Detection Camera")
            st.info("🎯 Position yourself in front of the camera for real-time emotion detection")
            
            try:
                from streamlit_webrtc import webrtc_streamer
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                webrtc_streamer(
                    key="emotion_detect", 
                    desired_playing_state=True, 
                    video_processor_factory=EmotionProcessor,
                    media_stream_constraints={"video": True, "audio": False}
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except ImportError:
                st.warning("⚠️ WebRTC not available. Install streamlit-webrtc to enable camera functionality.")
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("🔄 Reset Detection", use_container_width=True):
                np.save("emotion.npy", np.array([""]))
                st.success("✅ Emotion detection reset!")
                st.rerun()
        with col_reset2:
            if st.button("🧹 Clear Preferences", use_container_width=True):
                st.session_state['pref_lang'] = ''
                st.session_state['pref_singer'] = ''
                st.success("✅ Preferences cleared!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Music Platforms")
        
        # Platform buttons
        if current_emotion and lang and singer:
            query = f"{lang} {current_emotion} song {singer}".replace(" ", "+")
            
            platforms = [
                {"name": "YouTube", "icon": "🔴", "url": f"https://www.youtube.com/results?search_query={query}"},
                {"name": "YouTube Music", "icon": "🎵", "url": f"https://music.youtube.com/search?q={query}"},
                {"name": "Spotify", "icon": "🟢", "url": f"https://open.spotify.com/search/{query}"},
                {"name": "Apple Music", "icon": "🎧", "url": f"https://music.apple.com/search?term={query}"},
                {"name": "SoundCloud", "icon": "🟠", "url": f"https://soundcloud.com/search?q={query}"},
                {"name": "Amazon Music", "icon": "🔵", "url": f"https://music.amazon.com/search/{query}"}
            ]
            
            st.markdown('<div class="platform-grid">', unsafe_allow_html=True)
            for platform in platforms:
                st.markdown(f"""
                <div class="platform-card" onclick="window.open('{platform['url']}', '_blank')">
                    <span class="platform-icon">{platform['icon']}</span>
                    <div class="platform-name">{platform['name']}</div>
                    <div class="platform-description">Find {current_emotion} {lang} music</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"{platform['icon']} Open {platform['name']}", key=f"platform_{platform['name']}", use_container_width=True):
                    webbrowser.open_new_tab(platform['url'])
                    st.success(f"🎵 Opening {platform['name']}!")
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.7);'>
                <h4 style='margin-bottom: 1rem;'>🔒 Music Platforms Locked</h4>
                <p>Complete your preferences and let AI detect your emotion to unlock all music streaming platforms!</p>
                <br>
                <p style='font-size: 0.9rem;'>📝 Step 1: Set language and artist preferences<br>
                🎭 Step 2: Allow camera access for emotion detection<br>
                🎵 Step 3: Discover personalized music!</p>
            </div>
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Quick Stats")
        col = get_history_collection()
        if col:
            try:
                total_sessions = col.count_documents({'username': username})
                recent_emotions = list(col.find({'username': username}).sort('timestamp', -1).limit(5))
                
                st.metric("Total Sessions", total_sessions, delta=None)
                
                if recent_emotions:
                    recent_emotion = recent_emotions[0]['emotion']
                    st.metric("Last Emotion", recent_emotion, delta=None)
                    
                    unique_emotions = len(set([e['emotion'] for e in recent_emotions]))
                    st.metric("Recent Variety", f"{unique_emotions} emotions", delta=None)
                else:
                    st.info("Start using emotion detection to see your stats!")
                    
            except Exception as e:
                st.info("Stats will appear after using the app")
        else:
            st.info("Database connection needed for stats")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ GAMES PAGE ------------------
elif st.session_state.current_page == 'Games':
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">🎮 Gaming Zone</h1>
            <p class="hero-subtitle">Take a break and enjoy some amazing games while your emotions recharge!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌟 Featured Games")
    st.markdown("Choose from our curated selection of popular games to play instantly in your browser!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display games in a modern grid
    st.markdown('<div class="games-grid">', unsafe_allow_html=True)
    
    # Create game cards using Streamlit columns
    for i in range(0, len(GAMES), 3):
        cols = st.columns(3)
        for j, game in enumerate(GAMES[i:i+3]):
            with cols[j]:
                st.markdown(f"""
                <div class="game-card">
                    <img src="{game['image']}" alt="{game['name']}" class="game-image">
                    <div class="game-content">
                        <div class="game-title">{game['name']}</div>
                        <div class="game-description">{game['description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"▶️ Play {game['name']}", key=f"game_{i+j}", use_container_width=True):
                    webbrowser.open_new_tab(game['url'])
                    st.success(f"🎮 Opening {game['name']}!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Browse all games section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🌐 Explore More Games")
    st.markdown("Discover thousands of free games on Poki - from action and adventure to puzzle and racing games!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎲 All Games", use_container_width=True):
            webbrowser.open_new_tab("https://poki.com/")
            st.success("🌐 Opening Poki Games!")
    with col2:
        if st.button("🏃 Action Games", use_container_width=True):
            webbrowser.open_new_tab("https://poki.com/en/action")
            st.success("⚡ Opening Action Games!")
    with col3:
        if st.button("🧩 Puzzle Games", use_container_width=True):
            webbrowser.open_new_tab("https://poki.com/en/puzzle")
            st.success("🧠 Opening Puzzle Games!")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ HISTORY PAGE ------------------
elif st.session_state.current_page == 'History':
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">📜 Emotion History</h1>
            <p class="hero-subtitle">Track your emotional journey and discover patterns in your mood over time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col = get_history_collection()
    if col is None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.error("🔌 Database connection not available")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fetch recent history
        docs = list(col.find({'username': username}).sort('timestamp', -1).limit(100))
        
        if not docs:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; padding: 3rem; color: rgba(255, 255, 255, 0.8);'>
                <h3>📝 No History Found</h3>
                <p>Start using emotion detection to see your journey here!</p>
                <br>
                <p style='font-size: 0.9rem;'>Your emotional patterns and music preferences will be tracked and displayed here.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Convert to DataFrame
            df = pd.DataFrame(docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Summary stats
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(docs)}</div>
                    <div class="stat-label">Total Sessions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                unique_emotions = df['emotion'].nunique()
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{unique_emotions}</div>
                    <div class="stat-label">Unique Emotions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                most_common = df['emotion'].mode()[0] if len(df) > 0 else "None"
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{most_common}</div>
                    <div class="stat-label">Most Common</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                days_active = df['timestamp'].dt.date.nunique()
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{days_active}</div>
                    <div class="stat-label">Days Active</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent history table
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Recent Activity")
            
            # Filter and display options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                days_filter = st.selectbox("📅 Time Period", 
                                         ["Last 7 days", "Last 30 days", "Last 90 days", "All time"], 
                                         index=1)
            with col_filter2:
                emotion_filter = st.selectbox("🎭 Emotion Filter", 
                                            ["All emotions"] + list(df['emotion'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if days_filter != "All time":
                days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
                cutoff_date = datetime.now() - timedelta(days=days_map[days_filter])
                filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_date]
            
            if emotion_filter != "All emotions":
                filtered_df = filtered_df[filtered_df['emotion'] == emotion_filter]
            
            # Display filtered data
            if len(filtered_df) > 0:
                display_df = filtered_df[['timestamp', 'emotion', 'language', 'singer']].head(50)
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df.columns = ['Date & Time', 'Emotion', 'Language', 'Singer']
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No data found for the selected filters.")
            
            # Download section
            st.markdown("---")
            col_download1, col_download2 = st.columns(2)
            with col_download1:
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full History (CSV)",
                    data=csv,
                    file_name=f"emotion_history_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_download2:
                # Download filtered data
                if len(filtered_df) > 0:
                    filtered_csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data (CSV)",
                        data=filtered_csv,
                        file_name=f"filtered_emotions_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ ANALYTICS PAGE ------------------
elif st.session_state.current_page == 'Analytics':
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">📊 Analytics Dashboard</h1>
            <p class="hero-subtitle">Understand your emotional patterns, music preferences, and discover insights about your mood</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col = get_history_collection()
    if col is None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.error("🔌 Database connection not available")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        docs = list(col.find({'username': username}))
        
        if not docs:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; padding: 3rem; color: rgba(255, 255, 255, 0.8);'>
                <h3>📈 No Analytics Data</h3>
                <p>Start using the emotion detection feature to see powerful insights about your emotional patterns!</p>
                <br>
                <p style='font-size: 0.9rem;'>Analytics will include emotion distribution, timing patterns, music preferences, and much more.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            df = pd.DataFrame(docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Overview metrics
            total_sessions = len(df)
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            avg_daily = total_sessions / max(date_range, 1) if date_range > 0 else total_sessions
            
            # Time-based analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### 📈 Emotion Distribution")
                emotion_counts = df['emotion'].value_counts()
                fig = px.pie(
                    values=emotion_counts.values, 
                    names=emotion_counts.index,
                    title="Your Emotional Patterns",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### 📅 Daily Activity")
                daily_counts = df.set_index('timestamp').resample('D').size().reset_index()
                daily_counts.columns = ['Date', 'Count']
                fig = px.line(
                    daily_counts, 
                    x='Date', 
                    y='Count',
                    title="Daily Emotion Detection Sessions",
                    color_discrete_sequence=['#4facfe'],
                    markers=True
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400
                )
                fig.update_traces(line_color='#4facfe', marker_color='#00f2fe')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed patterns
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 🕐 Hourly Emotion Patterns")
            df['hour'] = df['timestamp'].dt.hour
            hourly_patterns = df.groupby(['hour', 'emotion']).size().reset_index(name='count')
            fig = px.bar(
                hourly_patterns, 
                x='hour', 
                y='count', 
                color='emotion',
                title="When do you feel different emotions throughout the day?",
                color_discrete_sequence=px.colors.qualitative.Set3,
                barmode='stack'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Hour of Day (24h format)",
                yaxis_title="Number of Sessions",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Weekly patterns
            col3, col4 = st.columns(2)
            with col3:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### 📅 Weekly Patterns")
                df['weekday'] = df['timestamp'].dt.day_name()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_patterns = df.groupby(['weekday', 'emotion']).size().reset_index(name='count')
                fig = px.bar(
                    weekly_patterns, 
                    x='weekday', 
                    y='count', 
                    color='emotion',
                    title="Emotional patterns by day of week",
                    category_orders={'weekday': weekday_order},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### 📈 Emotion Trends")
                df['date'] = df['timestamp'].dt.date
                emotion_trends = df.groupby(['date', 'emotion']).size().unstack(fill_value=0).reset_index()
                
                # Create trend chart for top 3 emotions
                top_emotions = df['emotion'].value_counts().head(3).index.tolist()
                fig = go.Figure()
                
                colors = ['#4facfe', '#f093fb', '#a8edea']
                for i, emotion in enumerate(top_emotions):
                    if emotion in emotion_trends.columns:
                        fig.add_trace(go.Scatter(
                            x=emotion_trends['date'], 
                            y=emotion_trends[emotion],
                            mode='lines+markers',
                            name=emotion,
                            line_color=colors[i % len(colors)]
                        ))
                
                fig.update_layout(
                    title="Emotion trends over time",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Music preferences analysis
            if 'language' in df.columns and 'singer' in df.columns:
                col5, col6 = st.columns(2)
                
                with col5:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("### 🌍 Language Preferences")
                    lang_counts = df[df['language'].notna() & (df['language'] != '')]['language'].value_counts().head(10)
                    if len(lang_counts) > 0:
                        fig = px.bar(
                            x=lang_counts.index, 
                            y=lang_counts.values,
                            title="Most searched music languages",
                            color_discrete_sequence=['#764ba2'],
                            labels={'x': 'Language', 'y': 'Search Count'}
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No language data available yet")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col6:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("### 🎤 Favorite Artists")
                    singer_counts = df[df['singer'].notna() & (df['singer'] != '')]['singer'].value_counts().head(10)
                    if len(singer_counts) > 0:
                        fig = px.bar(
                            x=singer_counts.values,
                            y=singer_counts.index, 
                            orientation='h',
                            title="Most searched artists",
                            color_discrete_sequence=['#667eea'],
                            labels={'x': 'Search Count', 'y': 'Artist'}
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No artist data available yet")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced analytics
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Insights")
            
            # Calculate insights
            most_active_hour = df['hour'].mode()[0] if len(df) > 0 else "Unknown"
            most_active_day = df['weekday'].mode()[0] if len(df) > 0 else "Unknown"
            dominant_emotion = df['emotion'].mode()[0] if len(df) > 0 else "Unknown"
            
            # Emotional diversity score
            emotion_diversity = len(df['emotion'].unique()) / len(df) * 100 if len(df) > 0 else 0
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: rgba(79, 172, 254, 0.1); border-radius: 12px; border: 1px solid rgba(79, 172, 254, 0.3);'>
                    <h4 style='color: #4facfe; margin-bottom: 0.5rem;'>⏰ Peak Activity</h4>
                    <p style='color: white; font-size: 1.1rem; margin: 0;'>{most_active_hour}:00 on {most_active_day}s</p>
                    <small style='color: rgba(255,255,255,0.7);'>Your most active time</small>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: rgba(240, 147, 251, 0.1); border-radius: 12px; border: 1px solid rgba(240, 147, 251, 0.3);'>
                    <h4 style='color: #f093fb; margin-bottom: 0.5rem;'>🎭 Dominant Emotion</h4>
                    <p style='color: white; font-size: 1.1rem; margin: 0;'>{dominant_emotion.title()}</p>
                    <small style='color: rgba(255,255,255,0.7);'>Your most frequent mood</small>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_col3:
                diversity_color = "#a8edea" if emotion_diversity > 15 else "#f5576c" if emotion_diversity < 5 else "#fed6e3"
                diversity_label = "High" if emotion_diversity > 15 else "Low" if emotion_diversity < 5 else "Medium"
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: rgba(168, 237, 234, 0.1); border-radius: 12px; border: 1px solid rgba(168, 237, 234, 0.3);'>
                    <h4 style='color: {diversity_color}; margin-bottom: 0.5rem;'>🌈 Emotional Range</h4>
                    <p style='color: white; font-size: 1.1rem; margin: 0;'>{diversity_label} ({emotion_diversity:.1f}%)</p>
                    <small style='color: rgba(255,255,255,0.7);'>Emotional diversity score</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations")
            
            recommendations = []
            
            if emotion_diversity < 10:
                recommendations.append("🎭 Try exploring music during different emotional states to discover new genres!")
            
            if most_active_hour in [22, 23, 0, 1, 2]:
                recommendations.append("🌙 You're often active late at night - consider calming music for better sleep!")
            
            if most_active_day in ['Saturday', 'Sunday']:
                recommendations.append("🎉 Weekend warrior! Your music taste might be more adventurous on weekends.")
            
            if len(recommendations) == 0:
                recommendations.append("✨ You have great emotional awareness! Keep exploring music that matches your moods.")
            
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div style='padding: 1rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #4facfe;'>
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PROFILE PAGE ------------------
elif st.session_state.current_page == 'Profile':
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1 class="hero-title">👤 User Profile</h1>
            <p class="hero-subtitle">Manage your account settings, preferences, and view your musical journey</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    db = get_database()
    if db is None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.error("🔌 Database connection not available")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        users_collection = db['users']
        prefs_collection = get_user_preferences_collection()
        
        user_doc = users_collection.find_one({'username': username})
        
        if not user_doc:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.error("👤 User profile not found")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Profile header
            st.markdown(f"""
            <div class="profile-header">
                <div class="profile-avatar-large">
                    {username[0].upper()}
                </div>
                <div class="profile-info">
                    <h2>{user_doc.get('full_name', username)}</h2>
                    <p>{user_doc.get('email', 'No email provided')} • Member since {user_doc.get('created_at', datetime.now()).strftime('%B %Y') if user_doc.get('created_at') else 'Unknown'}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Account Details")
                
                st.markdown(f"""
                <div style='padding: 1rem 0;'>
                    <div style='margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);'>
                        <strong style='color: #4facfe;'>Username</strong><br>
                        <span style='color: white;'>{user_doc.get('username')}</span>
                    </div>
                    <div style='margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);'>
                        <strong style='color: #4facfe;'>Email</strong><br>
                        <span style='color: white;'>{user_doc.get('email', 'Not provided')}</span>
                    </div>
                    <div style='margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);'>
                        <strong style='color: #4facfe;'>Phone</strong><br>
                        <span style='color: white;'>{user_doc.get('phone', 'Not provided')}</span>
                    </div>
                    <div style='margin-bottom: 1rem;'>
                        <strong style='color: #4facfe;'>Member Since</strong><br>
                        <span style='color: white;'>{user_doc.get('created_at', datetime.now()).strftime('%B %d, %Y') if user_doc.get('created_at') else 'Unknown'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Quick actions
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### ⚡ Quick Actions")
                
                if st.button("🔄 Reset All Data", use_container_width=True):
                    # Clear emotion file
                    np.save("emotion.npy", np.array([""]))
                    # Clear session state
                    if 'pref_lang' in st.session_state:
                        del st.session_state['pref_lang']
                    if 'pref_singer' in st.session_state:
                        del st.session_state['pref_singer']
                    st.success("✅ All data reset successfully!")
                    st.rerun()
                
                if st.button("🧹 Clear History", use_container_width=True):
                    col = get_history_collection()
                    if col:
                        result = col.delete_many({'username': username})
                        st.success(f"✅ Deleted {result.deleted_count} history records!")
                        st.rerun()
                    else:
                        st.error("Could not access history collection")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### ⚙️ Edit Profile")
                
                # Get current preferences
                current_prefs = prefs_collection.find_one({'username': username}) if prefs_collection else {}
                
                with st.form("profile_form"):
                    st.markdown("#### 👤 Personal Information")
                    new_full_name = st.text_input("Full Name", value=user_doc.get('full_name', ''))
                    new_email = st.text_input("Email", value=user_doc.get('email', ''))
                    new_phone = st.text_input("Phone", value=user_doc.get('phone', ''))
                    
                    st.markdown("#### 🎵 Music Preferences")
                    default_lang = st.text_input("Default Language", 
                                                value=current_prefs.get('default_language', ''),
                                                placeholder="e.g., English, Hindi, Spanish")
                    favorite_genres = st.text_input("Favorite Genres", 
                                                   value=current_prefs.get('favorite_genres', ''),
                                                   placeholder="e.g., Pop, Rock, Classical, Jazz")
                    favorite_artists = st.text_area("Favorite Artists", 
                                                    value=current_prefs.get('favorite_artists', ''),
                                                    placeholder="List your favorite artists, separated by commas")
                    
                    st.markdown("#### 🎭 Emotion Settings")
                    auto_detect = st.checkbox("Auto-detect emotions on app start", 
                                            value=current_prefs.get('auto_detect', False))
                    save_history = st.checkbox("Save emotion history", 
                                             value=current_prefs.get('save_history', True))
                    
                    col_submit1, col_submit2 = st.columns(2)
                    with col_submit1:
                        submit_profile = st.form_submit_button("💾 Save Changes", use_container_width=True)
                    with col_submit2:
                        cancel_changes = st.form_submit_button("❌ Cancel", use_container_width=True)
                    
                    if submit_profile:
                        try:
                            # Update user profile
                            update_data = {
                                'full_name': new_full_name,
                                'email': new_email,
                                'phone': new_phone,
                                'updated_at': datetime.utcnow()
                            }
                            
                            users_collection.update_one(
                                {'username': username},
                                {'$set': update_data}
                            )
                            
                            # Update preferences
                            if prefs_collection:
                                prefs_data = {
                                    'username': username,
                                    'default_language': default_lang,
                                    'favorite_genres': favorite_genres,
                                    'favorite_artists': favorite_artists,
                                    'auto_detect': auto_detect,
                                    'save_history': save_history,
                                    'updated_at': datetime.utcnow()
                                }
                                
                                prefs_collection.update_one(
                                    {'username': username},
                                    {'$set': prefs_data},
                                    upsert=True
                                )
                            
                            st.success("✅ Profile updated successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error updating profile: {str(e)}")
                    
                    elif cancel_changes:
                        st.info("Changes cancelled")
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Account statistics
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Account Statistics")
            col = get_history_collection()
            if col:
                try:
                    total_sessions = col.count_documents({'username': username})
                    first_session = col.find_one({'username': username}, sort=[('timestamp', 1)])
                    last_session = col.find_one({'username': username}, sort=[('timestamp', -1)])
                    
                    # Calculate streaks and additional stats
                    recent_sessions = list(col.find({'username': username}).sort('timestamp', -1).limit(30))
                    unique_emotions_30d = len(set([s['emotion'] for s in recent_sessions])) if recent_sessions else 0
                    
                    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{total_sessions}</div>
                            <div class="stat-label">Total Sessions</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if first_session:
                            days_using = (datetime.utcnow() - first_session['timestamp']).days + 1
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{days_using}</div>
                                <div class="stat-label">Days Using App</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="stat-card">
                                <div class="stat-value">0</div>
                                <div class="stat-label">Days Using App</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if last_session:
                            last_used = last_session['timestamp'].strftime('%m/%d')
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{last_used}</div>
                                <div class="stat-label">Last Used</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="stat-card">
                                <div class="stat-value">Never</div>
                                <div class="stat-label">Last Used</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{unique_emotions_30d}</div>
                            <div class="stat-label">Emotions (30d)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Achievement badges
                    st.markdown("---")
                    st.markdown("### 🏆 Achievements")
                    
                    achievements = []
                    if total_sessions >= 1:
                        achievements.append({"name": "First Steps", "icon": "👶", "desc": "Used emotion detection for the first time"})
                    if total_sessions >= 10:
                        achievements.append({"name": "Explorer", "icon": "🗺️", "desc": "Completed 10 emotion detection sessions"})
                    if total_sessions >= 50:
                        achievements.append({"name": "Enthusiast", "icon": "🌟", "desc": "Reached 50 sessions milestone"})
                    if total_sessions >= 100:
                        achievements.append({"name": "Master", "icon": "🎓", "desc": "Achieved 100 sessions - emotion detection master!"})
                    if unique_emotions_30d >= 5:
                        achievements.append({"name": "Emotional Range", "icon": "🎭", "desc": "Expressed 5+ different emotions in 30 days"})
                    if days_using >= 7:
                        achievements.append({"name": "Week Warrior", "icon": "📅", "desc": "Used the app for a full week"})
                    if days_using >= 30:
                        achievements.append({"name": "Monthly Master", "icon": "🗓️", "desc": "Active user for 30+ days"})
                    
                    if achievements:
                        achievement_cols = st.columns(min(4, len(achievements)))
                        for i, achievement in enumerate(achievements[:8]):  # Show max 8 achievements
                            with achievement_cols[i % 4]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.1);'>
                                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{achievement['icon']}</div>
                                    <div style='color: #4facfe; font-weight: 600; margin-bottom: 0.25rem;'>{achievement['name']}</div>
                                    <div style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>{achievement['desc']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("🎯 Start using emotion detection to unlock achievements!")
                    
                except Exception as e:
                    st.info("Statistics will appear after using the app more")
            else:
                st.info("Database connection needed for statistics")
            st.markdown('</div>', unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("""
<div style='background: rgba(0, 0, 0, 0.2); backdrop-filter: blur(20px); border-top: 1px solid rgba(255, 255, 255, 0.1); padding: 3rem 2rem; text-align: center; color: rgba(255, 255, 255, 0.8); margin-top: 4rem;'>
    <div style='max-width: 1200px; margin: 0 auto;'>
        <h3 style='color: white; margin-bottom: 1rem; font-size: 1.5rem;'>🎵 MoodBeats</h3>
        <p style='margin-bottom: 2rem; line-height: 1.6;'>AI-powered music discovery that understands your emotions and connects you with the perfect soundtrack for every moment</p>
        
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin: 2rem 0;'>
            <div>
                <h4 style='color: #4facfe; margin-bottom: 0.5rem;'>🎭 Features</h4>
                <p style='font-size: 0.9rem;'>Real-time emotion detection<br>Multi-platform music search<br>Detailed analytics & insights</p>
            </div>
            <div>
                <h4 style='color: #f093fb; margin-bottom: 0.5rem;'>🎮 Entertainment</h4>
                <p style='font-size: 0.9rem;'>Integrated gaming platform<br>Popular browser games<br>Mood-lifting activities</p>
            </div>
            <div>
                <h4 style='color: #a8edea; margin-bottom: 0.5rem;'>📊 Analytics</h4>
                <p style='font-size: 0.9rem;'>Emotion pattern tracking<br>Music preference insights<br>Personal recommendations</p>
            </div>
        </div>
        
        <div style='border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 2rem; margin-top: 2rem;'>
            <p style='font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.5rem;'>Built with ❤️ using Streamlit • MediaPipe • TensorFlow • Plotly</p>
            <p style='font-size: 0.85rem; opacity: 0.7;'>Gaming powered by Poki • Music platforms integrated for seamless discovery</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)