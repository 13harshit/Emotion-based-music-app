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
import time
from urllib.parse import quote_plus

# Import auth functions
from auth import is_authenticated, show_auth_page, logout
from database import db_manager
## Removed: from music_recommendation import integrate_enhanced_recommendations

# Optional speech recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except Exception:
    SPEECH_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AI Music Player + Games", 
    page_icon="🎵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .game-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-align: center;
        overflow: hidden;
    }
    
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .game-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
    }
    
    .emotion-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .platform-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        margin: 0.5rem;
        width: 100%;
    }
    
    .platform-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .video-container {
        border: 3px solid #667eea;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar-metric {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Helper Functions ------------------
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("model.h5")
        
        # Try to load labels with different methods
        try:
            # First try without allow_pickle
            labels = np.load("labels.npy", allow_pickle=False)
        except:
            try:
                # If that fails, try with allow_pickle
                labels = np.load("labels.npy", allow_pickle=True)
            except:
                # If both fail, create default labels
                st.warning("Could not load labels.npy, using default emotion labels")
                labels = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
                # Save the default labels for future use
                np.save("labels.npy", labels, allow_pickle=False)
        
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
    db = db_manager.db
    return db['emotion_history'] if db is not None else None

def get_user_preferences_collection():
    db = db_manager.db
    return db['user_preferences'] if db is not None else None

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
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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
                              landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                              connection_drawing_spec=drawing.DrawingSpec(thickness=1))
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
    except AttributeError as e:
        return None, "Microphone not found or not working. Please check your device and permissions."
    except Exception as e:
        return None, f"Error: {str(e)}"

# ------------------ Authentication Check ------------------
if not is_authenticated():
    show_auth_page()
    st.stop()

username = st.session_state.get('username', 'Unknown User')

# ------------------ Sidebar Navigation ------------------
with st.sidebar:
    st.markdown("### 🎵 Navigation")
    nav = st.radio("", ["🏠 Home", "🎮 Games", "📜 History", "📊 Analytics", "👤 Profile"])
    
    st.markdown("---")
    st.markdown(f"### Welcome, **{username}**!")
    
    # Quick stats in sidebar
    try:
        col = get_history_collection()
        if col is not None:
            total_detections = col.count_documents({'username': username})
            unique_emotions = len(col.distinct('emotion', {'username': username}))
            last_session = col.find_one({'username': username}, sort=[('timestamp', -1)])
            
            st.markdown(f"""
            <div class="sidebar-metric">
                <h4>📊 Your Stats</h4>
                <p><strong>{total_detections}</strong> Total Detections</p>
                <p><strong>{unique_emotions}</strong> Emotions Detected</p>
            </div>
            """, unsafe_allow_html=True)
            
            if last_session:
                last_emotion = last_session.get('emotion', 'Unknown')
                st.markdown(f"""
                <div class="sidebar-metric">
                    <h4>🎭 Last Emotion</h4>
                    <p><strong>{last_emotion}</strong></p>
                </div>
                """, unsafe_allow_html=True)
    except:
        pass
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        logout()

# ------------------ Game Data ------------------
import base64
import os

# Function to encode local images to base64
def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

GAMES = [
    {
        "name": "Geometry Dash",
        "image": "images/Geometry_Dash.jpg",
        "url": "https://poki.com/en/g/geometry-dash-lite",
        "description": "Jump and fly through danger!"
    },
    {
        "name": "Subway Surfers",
        "image": "images/Subway_Surfers.jpg",
        "url": "https://poki.com/en/g/subway-surfers",
        "description": "Run as far as you can!"
    },
    {
        "name": "Temple Run 2",
        "image": "images/Temple_Run _2.jpg",
        "url": "https://poki.com/en/g/temple-run-2",
        "description": "Navigate perilous cliffs!"
    },
    {
        "name": "Cut the Rope",
        "image": "images/Cut_the_Rope.jpg",
        "url": "https://poki.com/en/g/cut-the-rope",
        "description": "Feed candy to Om Nom!"
    },
    {
        "name": "Moto X3M",
        "image": "images/Moto_X3M.jpg",
        "url": "https://poki.com/en/g/moto-x3m",
        "description": "Bike racing with stunts!"
    },
    {
        "name": "Fireboy and Watergirl",
        "image": "images/Fireboy_and_Watergirl.jpeg",
        "url": "https://poki.com/en/g/fireboy-and-water-girl-the-forest-temple",
        "description": "Teamwork puzzle adventure!"
    }
]

# ------------------ HOME PAGE ------------------
if nav == "🏠 Home":
    # Removed: integrate_enhanced_recommendations()
    
    st.markdown("""
    <div class="main-header">
        <h1>🎵 AI-Powered Music Player</h1>
        <p>Discover music that matches your emotions with AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current emotion status
    try:
        current_emotion = np.load("emotion.npy", allow_pickle=True)[0]
    except:
        current_emotion = ""
    
    if current_emotion:
        st.markdown(f"""
        <div class="emotion-display">
            🎭 Current Detected Emotion: {current_emotion.upper()}
        </div>
        """, unsafe_allow_html=True)
    
    # Main features in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎼 Music Preferences")
        
        # Voice input section
        if SPEECH_AVAILABLE:
            voice_col1, voice_col2 = st.columns([3, 1])
            with voice_col2:
                if st.button("🎤 Voice Input"):
                    with st.spinner("Listening..."):
                        text, status = speech_to_text()
                    if text:
                        st.success(f"Heard: {text}")
                        # Simple parsing - you can enhance this
                        if "language" in text.lower() or "lang" in text.lower():
                            st.session_state['pref_lang'] = text.split()[-1]
                        elif "singer" in text.lower() or "artist" in text.lower():
                            st.session_state['pref_singer'] = text.split()[-1]
                        else:
                            # Try to parse "Hindi song by Arijit Singh"
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
            "🌍 Language", 
            placeholder="e.g., Hindi, English, Spanish",
            value=st.session_state.get('pref_lang', '')
        )
        singer = st.text_input(
            "🎤 Singer/Artist", 
            placeholder="e.g., Arijit Singh, Taylor Swift",
            value=st.session_state.get('pref_singer', '')
        )
        
        st.session_state['pref_lang'] = lang
        st.session_state['pref_singer'] = singer
        
        # Camera section
        # Camera section: Show only if emotion is NOT detected
        if not current_emotion and lang and singer:
            st.markdown(" Emotion Detection")
            st.info("Position yourself in front of the camera for emotion detection")
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
                st.error("Camera functionality requires streamlit-webrtc package. Please install it with: pip install streamlit-webrtc")
            except Exception as e:
                st.error(f"Camera error: {e}")
                st.info("Please check camera permissions and try refreshing the page")

    with col2:
        st.markdown(" Quick Actions")
        
        # Platform buttons
        if current_emotion and lang and singer:
            # Enhanced query with emotion keywords
            emotion_keywords = {
                'happy': 'upbeat energetic',
                'sad': 'emotional slow',
                'angry': 'intense rock',
                'neutral': 'popular hits',
                'surprise': 'dance upbeat',
                'fear': 'calming soft',
                'disgust': 'alternative indie'
            }
            
            base_query = f"{lang} {singer}"
            emotion_boost = emotion_keywords.get(current_emotion, current_emotion)
            enhanced_query = f"{base_query} {emotion_boost}".replace(" ", "+")
            
            # Primary platforms
            platforms = [
                ("🔴 YouTube", f"https://www.youtube.com/results?search_query={enhanced_query}"),
                ("🎵 YT Music", f"https://music.youtube.com/search?q={enhanced_query}"),
                ("🟢 Spotify", f"https://open.spotify.com/search/{enhanced_query}"),
                ("🎧 Apple Music", f"https://music.apple.com/search?term={enhanced_query}")
            ]
            
            for name, url in platforms:
                if st.button(name, key=f"btn_{name.replace(' ', '_').lower()}", use_container_width=True):
                    webbrowser.open_new_tab(url)
                    st.success(f"🎉 Opening {name}!")
                    st.balloons()
                    
                    # Save to database if available
                    try:
                        from database import save_music_recommendation
                        save_music_recommendation(
                            username=st.session_state.get('username', 'anonymous'),
                            platform=name.split()[-1].lower(),
                            query=enhanced_query.replace('+', ' '),
                            emotion=current_emotion,
                            language=lang,
                            artist=singer
                        )
                    except:
                        pass
                    
                    # Clear emotion
                    time.sleep(1)
                    np.save("emotion.npy", np.array([""]))
                    st.rerun()
            
            # More platforms in expander
            with st.expander("🌟 More Platforms"):
                more_platforms = [
                    ("🔊 SoundCloud", f"https://soundcloud.com/search?q={enhanced_query}"),
                    ("📦 Amazon Music", f"https://music.amazon.com/search/{enhanced_query}"),
                    ("📻 Pandora", f"https://www.pandora.com/search/{enhanced_query}"),
                    ("🎶 Deezer", f"https://www.deezer.com/search/{enhanced_query}")
                ]
                
                for name, url in more_platforms:
                    if st.button(name, key=f"more_{name.replace(' ', '_').lower()}", use_container_width=True):
                        webbrowser.open_new_tab(url)
                        st.success(f"Opening {name}!")
            
            # Main recommendation button (enhanced version of your existing one)
            st.markdown("---")
            if st.button("� Get Perfect Music Match", key="perfect_match", use_container_width=True, type="primary"):
                with st.spinner("🔍 Finding perfect music..."):
                    # Your original logic but enhanced
                    smart_query = f"{lang}+{current_emotion}+song+{singer}+{emotion_keywords.get(current_emotion, '')}"
                    webbrowser.open_new_tab(f"https://www.youtube.com/results?search_query={smart_query}")
                    
                    st.success(f"🎉 Found {current_emotion} songs by {singer} in {lang}!")
                    st.balloons()
                    
                    # Clear emotion
                    np.save("emotion.npy", np.array([""]))
                    time.sleep(1)
                    st.rerun()
        
        else:
            st.warning("Complete preferences and emotion detection to unlock music platforms!")
            
            # Show what's missing
            missing = []
            if not current_emotion:
                missing.append("🎭 Emotion")
            if not lang:
                missing.append("🌍 Language")
            if not singer:
                missing.append("🎤 Artist")
            
            for item in missing:
                st.markdown(f"❌ {item}")
        
        # Reset button (keep your existing one but enhance it)
        if st.button("🔄 Reset Detection", use_container_width=True):
            np.save("emotion.npy", np.array([""]))
            st.success("Emotion detection reset!")
            st.rerun()

# ------------------ GAMES PAGE ------------------
elif nav == "🎮 Games":
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Gaming Zone</h1>
        <p>Take a break and enjoy some games!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Popular Games")
    
    # Display games in a grid
    cols = st.columns(3)
    for i, game in enumerate(GAMES):
        with cols[i % 3]:
            # Create a complete game card with image, title, and description
            try:
                if os.path.exists(game['image']):
                    # Use base64 encoding to embed image in HTML
                    import base64
                    with open(game['image'], "rb") as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode()
                    
                    # Determine image format
                    img_format = "jpeg" if game['image'].endswith(('.jpg', '.jpeg')) else "png"
                    
                    st.markdown(f"""
                    <div class="game-card" style="padding: 0; overflow: hidden;">
                        <img src="data:image/{img_format};base64,{img_b64}" 
                             style="width: 100%; height: 200px; object-fit: cover; border-radius: 12px 12px 0 0;">
                        <div style="padding: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{game['name']}</h4>
                            <p style="color: #666; margin: 0; font-size: 14px;">{game['description']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Fallback design if image doesn't exist
                    st.markdown(f"""
                    <div class="game-card" style="padding: 0; overflow: hidden;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    height: 200px; display: flex; align-items: center; 
                                    justify-content: center; color: white; font-size: 24px;">
                            🎮
                        </div>
                        <div style="padding: 1rem;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{game['name']}</h4>
                            <p style="color: #666; margin: 0; font-size: 14px;">{game['description']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                # Error fallback
                st.markdown(f"""
                <div class="game-card" style="padding: 0; overflow: hidden;">
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                                height: 200px; display: flex; align-items: center; 
                                justify-content: center; color: white; font-size: 16px; text-align: center;">
                        ⚠️<br>Image Error
                    </div>
                    <div style="padding: 1rem;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #333;">{game['name']}</h4>
                        <p style="color: #666; margin: 0; font-size: 14px;">{game['description']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Play button
            if st.button(f"▶️ Play {game['name']}", key=f"game_{i}", use_container_width=True):
                webbrowser.open_new_tab(game['url'])
                st.success(f"Opening {game['name']}!")
    
    st.markdown("### 🎲 More Games")
    if st.button("🌐 Browse All Games on Poki", use_container_width=True):
        webbrowser.open_new_tab("https://poki.com/")
        st.success("Opening Poki Games!")

# ------------------ HISTORY PAGE ------------------
elif nav == "📜 History":
    st.markdown("""
    <div class="main-header">
        <h1>📜 Emotion Detection History</h1>
        <p>Track your emotional journey through music</p>
    </div>
    """, unsafe_allow_html=True)
    
    col = get_history_collection()
    if col is None:
        st.error("Database connection not available")
    else:
        # Fetch recent history
        docs = list(col.find({'username': username}).sort('timestamp', -1).limit(100))
        
        if not docs:
            st.info("No history found. Start using emotion detection to see your history here!")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="main-header">
                    <h1>🎵 AI-Powered Music Player</h1>
                    <p>Discover music that matches your emotions with AI technology</p>
                </div>
                """, unsafe_allow_html=True)

                # Recommend button at the top
                st.markdown("### 🎼 Music Preferences")
                col1, col2 = st.columns(2)
                with col1:
                    lang = st.text_input("🌍 Language", placeholder="e.g., Hindi, English, Spanish")
                with col2:
                    singer = st.text_input("🎤 Singer/Artist", placeholder="e.g., Arijit Singh, Taylor Swift")

                # Load current emotion
                try:
                    emotion = np.load("emotion.npy", allow_pickle=True)[0]
                except:
                    emotion = ""

                # Recommend button at the very top
                recommend_btn_top = st.button("🎯 Recommend Music Based on My Emotion (Top)", use_container_width=True)
                if recommend_btn_top:
                    if not lang or not singer:
                        st.error("⚠️ Please fill in both language and singer preferences!")
                    elif not emotion:
                        st.error("⚠️ Please ensure your emotion is detected!")
                    else:
                        search_query = f"{lang}+{emotion}+song+{singer}"
                        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
                        st.success(f"🎉 Opening YouTube with {emotion} songs by {singer} in {lang}!")
            
            # Recent history table
            st.markdown("### 📋 Recent History")
            display_df = df[['timestamp', 'emotion', 'language', 'singer']].head(20)
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df.columns = ['Date & Time', 'Emotion', 'Language', 'Singer']
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"emotion_history_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ------------------ ANALYTICS PAGE ------------------
elif nav == "📊 Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>Advanced Analytics Dashboard</h1>
        <p>Understand your emotional patterns and music preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    col = get_history_collection()
    if col is None:
        st.error("Database connection not available")
    else:
        docs = list(col.find({'username': username}))
        
        if not docs:
            st.info("No data available for analytics. Start using the app to see insights!")
        else:
            df = pd.DataFrame(docs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time-based analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Emotion Frequency")
                emotion_counts = df['emotion'].value_counts()
                fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                           title="Distribution of Detected Emotions")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📅 Daily Activity")
                daily_counts = df.set_index('timestamp').resample('D').size().reset_index()
                daily_counts.columns = ['Date', 'Count']
                fig = px.line(daily_counts, x='Date', y='Count', 
                            title="Daily Emotion Detections")
                st.plotly_chart(fig, use_container_width=True)
            
            # Hourly patterns
            st.markdown("### 🕐 Hourly Patterns")
            df['hour'] = df['timestamp'].dt.hour
            hourly_patterns = df.groupby(['hour', 'emotion']).size().reset_index(name='count')
            fig = px.bar(hourly_patterns, x='hour', y='count', color='emotion',
                        title="Emotion Detection by Hour of Day")
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly patterns
            st.markdown("### 📅 Weekly Patterns")
            df['weekday'] = df['timestamp'].dt.day_name()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_patterns = df.groupby(['weekday', 'emotion']).size().reset_index(name='count')
            fig = px.bar(weekly_patterns, x='weekday', y='count', color='emotion',
                        title="Emotion Detection by Day of Week",
                        category_orders={'weekday': weekday_order})
            st.plotly_chart(fig, use_container_width=True)
            
            # Music preferences analysis
            if 'language' in df.columns and 'singer' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🌍 Language Preferences")
                    lang_counts = df['language'].value_counts().head(10)
                    fig = px.bar(x=lang_counts.index, y=lang_counts.values,
                               title="Most Searched Languages")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎤 Favorite Artists")
                    singer_counts = df['singer'].value_counts().head(10)
                    fig = px.bar(x=singer_counts.index, y=singer_counts.values,
                               title="Most Searched Artists")
                    st.plotly_chart(fig, use_container_width=True)

# ------------------ PROFILE PAGE ------------------
elif nav == "👤 Profile":
    st.markdown("""
    <div class="main-header">
        <h1>User Profile</h1>
        <p>Manage your account settings and preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    db = db_manager.db
    if db is None:
        st.error("Database connection not available")
    else:
        users_collection = db['users']
        prefs_collection = get_user_preferences_collection()
        
        user_doc = users_collection.find_one({'username': username})
        
        if not user_doc:
            st.error("User profile not found")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 📋 Account Information")
                st.info(f"**Username:** {user_doc.get('username')}")
                st.info(f"**Email:** {user_doc.get('email', 'Not provided')}")
                st.info(f"**Phone:** {user_doc.get('phone', 'Not provided')}")
                st.info(f"**Member Since:** {user_doc.get('created_at', 'Unknown').strftime('%Y-%m-%d') if user_doc.get('created_at') else 'Unknown'}")
            
            with col2:
                st.markdown("### ⚙️ Edit Profile")
                
                with st.form("profile_form"):
                    new_email = st.text_input("Email", value=user_doc.get('email', ''))
                    new_phone = st.text_input("Phone", value=user_doc.get('phone', ''))
                    
                    # Music preferences
                    st.markdown("#### 🎵 Music Preferences")
                    default_lang = st.text_input("Default Language", 
                                                placeholder="e.g., English, Hindi")
                    favorite_genres = st.text_input("Favorite Genres", 
                                                   placeholder="e.g., Pop, Rock, Classical")
                    favorite_artists = st.text_area("Favorite Artists", 
                                                    placeholder="List your favorite artists")
                    
                    if st.form_submit_button("💾 Save Changes", use_container_width=True):
                        # Update user profile
                        users_collection.update_one(
                            {'username': username},
                            {'$set': {
                                'email': new_email,
                                'phone': new_phone,
                                'updated_at': datetime.utcnow()
                            }}
                        )
                        
                        # Update preferences
                        if prefs_collection is not None:
                            prefs_collection.update_one(
                                {'username': username},
                                {'$set': {
                                    'default_language': default_lang,
                                    'favorite_genres': favorite_genres,
                                    'favorite_artists': favorite_artists,
                                    'updated_at': datetime.utcnow()
                                }},
                                upsert=True
                            )
                        
                        st.success("Profile updated successfully!")
                        st.rerun()
            
            # Account statistics
            st.markdown("### 📊 Account Statistics")
            col = get_history_collection()
            if col is not None:
                total_sessions = col.count_documents({'username': username})
                first_session = col.find_one({'username': username}, sort=[('timestamp', 1)])
                last_session = col.find_one({'username': username}, sort=[('timestamp', -1)])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sessions", total_sessions)
                with col2:
                    if first_session:
                        days_using = (datetime.utcnow() - first_session['timestamp']).days
                        st.metric("Days Using App", days_using)
                    else:
                        st.metric("Days Using App", 0)
                with col3:
                    if last_session:
                        last_used = last_session['timestamp'].strftime('%Y-%m-%d')
                        st.metric("Last Used", last_used)
                    else:
                        st.metric("Last Used", "Never")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem;'>"
    "<h4>AI Music Player</h4>"
    "<p>Discover music that matches your mood with advanced AI emotion detection</p>"
    "<p>Built with using Streamlit, MediaPipe, TensorFlow, and MongoDB</p>"
    "<p><em>Features: Multi-platform integration Voice-to-text  Advanced analytics  Gaming zone</em></p>"
    "</div>",
    unsafe_allow_html=True
)