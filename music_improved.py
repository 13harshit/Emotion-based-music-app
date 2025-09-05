import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
from images.auth import is_authenticated, logout, show_auth_page

# Page configuration
st.set_page_config(
    page_title="AI Music Player",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
    .user-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .input-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .video-container {
        border: 3px solid #1f77b4;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .logout-btn {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Check authentication
if not is_authenticated():
    show_auth_page()
    st.stop()

# Load ML model and components
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("model.h5")
        
        # Try to load labels with different methods
        try:
            # First try without allow_pickle
            label = np.load("labels.npy", allow_pickle=False)
        except:
            try:
                # If that fails, try with allow_pickle
                label = np.load("labels.npy", allow_pickle=True)
            except:
                # If both fail, create default labels
                st.warning("Could not load labels.npy, using default emotion labels")
                label = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
                # Save the default labels for future use
                np.save("labels.npy", label, allow_pickle=False)
        
        return model, label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, label = load_emotion_model()

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Try to read saved emotion
try:
    emotion = np.load("emotion.npy", allow_pickle=True)[0]
except:
    emotion = ""

# Update run state based on emotion
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Emotion processing class
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
                pred = label[np.argmax(model.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            
        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                            connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Sidebar
with st.sidebar:
    st.markdown("### 👤 User Dashboard")
    
    # User info
    username = st.session_state.get('username', 'Unknown User')
    st.markdown(f"""
    <div class="user-info">
        <h4>Welcome, {username}!</h4>
        <p>🎵 Ready to discover music based on your emotions?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Current emotion status
    if emotion:
        st.markdown(f"""
        <div class="emotion-display">
            Current Emotion: {emotion.upper()}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No emotion detected yet")
    
    # App statistics (placeholder)
    st.markdown("### 📊 Your Stats")
    st.metric("Songs Recommended", "42")
    st.metric("Emotions Detected", "7")
    st.metric("Sessions", "15")
    
    # Logout button
    if st.button("🚪 Logout", key="logout_btn"):
        logout()

# Main content
st.markdown('<h1 class="main-header">🎵 AI-Powered Music Player</h1>', unsafe_allow_html=True)

# Status section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="status-card">
        <h3>🤖 How it works</h3>
        <p>1. Allow camera access and position yourself in frame</p>
        <p>2. Enter your preferred language and singer</p>
        <p>3. Let AI analyze your facial expressions</p>
        <p>4. Get personalized music recommendations!</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if emotion:
        st.success(f"✅ Emotion detected: **{emotion}**")
    else:
        st.warning("⏳ Please position yourself in camera view")

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### 🎼 Music Preferences")

col1, col2 = st.columns(2)
with col1:
    lang = st.text_input("🌍 Language", placeholder="e.g., Hindi, English, Spanish")
with col2:
    singer = st.text_input("🎤 Singer/Artist", placeholder="e.g., Arijit Singh, Taylor Swift")

st.markdown('</div>', unsafe_allow_html=True)

# Camera section
if lang and singer and st.session_state["run"] != "false":
    st.markdown("### 📹 Emotion Detection Camera")
    st.info("Please stay still and look at the camera for accurate emotion detection")
    
    try:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        webrtc_streamer(
            key="emotion_detection",
            desired_playing_state=True,
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Camera error: {e}")
        st.info("Please check camera permissions and try refreshing the page")

# Recommendation section
st.markdown("### 🎵 Get Your Personalized Recommendations")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_btn = st.button("🎯 Recommend Music Based on My Emotion", use_container_width=True)

if recommend_btn:
    if not emotion:
        st.error("⚠️ Please ensure you're visible in the camera frame so I can detect your emotion!")
        st.session_state["run"] = "true"
    elif not lang or not singer:
        st.error("⚠️ Please fill in both language and singer preferences!")
    else:
        with st.spinner("🔍 Finding perfect songs for your mood..."):
            # Open YouTube search
            search_query = f"{lang}+{emotion}+song+{singer}"
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
            
            # Clear emotion and update state
            np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"
            
            st.success(f"🎉 Opening YouTube with {emotion} songs by {singer} in {lang}!")
            st.balloons()

# Additional features section
st.markdown("---")
st.markdown("### 🔧 Additional Features")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔄 Reset Emotion Detection"):
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "true"
        st.success("Emotion detection reset!")  # or recommendation success
        st.rerun()  # Only here, not in every frame

with col2:
    if st.button("📊 View Supported Emotions"):
        if label is not None:
            with st.expander("Supported Emotions"):
                for i, emotion_label in enumerate(label):
                    st.write(f"{i+1}. {emotion_label}")
        else:
            st.error("Model not loaded properly")

with col3:
    if st.button("ℹ️ Help & Tips"):
        with st.expander("Tips for better detection"):
            st.markdown("""
            - Ensure good lighting
            - Keep your face clearly visible
            - Stay still for a few seconds
            - Make sure camera permissions are granted
            - Try different facial expressions if needed
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎵 AI Music Player - Discover music that matches your mood!</p>
    <p>Built with ❤️ using Streamlit, MediaPipe, and TensorFlow</p>
</div>
""", unsafe_allow_html=True)
