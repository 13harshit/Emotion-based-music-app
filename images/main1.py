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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer
import time

# Import custom modules
from auth_enhanced import is_authenticated, show_auth_page, logout
from database import (
    get_emotion_history, save_emotion_detection, get_user_profile,
    update_user_preferences, get_user_analytics
)
from analytics import EmotionAnalytics
from music_platforms import MusicPlatforms
from games import GamesIntegration
from voice_handler import VoiceHandler

# Page configuration
st.set_page_config(
    page_title="Enhanced AI Music Player",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .game-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }
    
    .game-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .platform-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .platform-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .voice-button {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .video-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        min-width: 150px;
    }
    
    .navigation-button {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .navigation-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    .navigation-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Load ML model and MediaPipe
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model and labels"""
    try:
        model = load_model("model.h5")
        labels = np.load("labels.npy")
        return model, labels
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None, None

# Initialize components
model, labels = load_emotion_model()
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Player"
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = ""
if "voice_handler" not in st.session_state:
    st.session_state.voice_handler = VoiceHandler()

# Emotion processing class
class EmotionProcessor:
    def __init__(self, username):
        self.username = username
        
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []
        
        if res.face_landmarks:
            # Extract face landmarks
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Extract left hand landmarks
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            # Extract right hand landmarks
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            # Make prediction
            lst = np.array(lst).reshape(1, -1)
            if model is not None and labels is not None:
                pred = labels[np.argmax(model.predict(lst))]
                
                # Display emotion on frame
                cv2.putText(frm, str(pred), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 0, 0), 2)
                
                # Save emotion to session state and database
                st.session_state.current_emotion = str(pred)
                save_emotion_detection(self.username, str(pred))

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                              landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), 
                                                                       thickness=-1, circle_radius=1),
                              connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

def show_navigation():
    """Show navigation sidebar"""
    st.sidebar.markdown("### 🎵 Navigation")
    
    pages = {
        "🎵 Player": "Player",
        "🎮 Games": "Games", 
        "📜 History": "History",
        "📊 Analytics": "Analytics",
        "👤 Profile": "Profile"
    }
    
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()

def show_user_info():
    """Show user information in sidebar"""
    username = st.session_state.get('username', 'Unknown User')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 User Info")
    st.sidebar.markdown(f"**Welcome, {username}!**")
    
    # Show current emotion
    if st.session_state.current_emotion:
        st.sidebar.success(f"Current Emotion: {st.session_state.current_emotion}")
    else:
        st.sidebar.info("No emotion detected")
    
    # Quick stats
    try:
        analytics = EmotionAnalytics(username)
        stats = analytics.get_quick_stats()
        
        st.sidebar.markdown("#### 📊 Quick Stats")
        st.sidebar.metric("Total Sessions", stats.get('total_sessions', 0))
        st.sidebar.metric("Emotions Detected", stats.get('unique_emotions', 0))
        st.sidebar.metric("Most Common", stats.get('most_common_emotion', 'None'))
        
    except Exception as e:
        st.sidebar.error(f"Stats error: {e}")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

def show_player_page():
    """Main music player page with emotion detection"""
    st.markdown("""
    <div class="main-header">
        <h1>🎵 AI-Powered Music Player</h1>
        <p>Discover music that matches your emotions</p>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.session_state.get('username')
    
    # Get user preferences
    user_profile = get_user_profile(username)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎼 Music Preferences")
        
        # Music preference inputs
        col_lang, col_artist = st.columns(2)
        with col_lang:
            language = st.text_input("🌍 Language", 
                                   value=user_profile.get('preferred_language', ''),
                                   placeholder="e.g., English, Hindi, Spanish")
        with col_artist:
            artist = st.text_input("🎤 Artist/Singer", 
                                 value=user_profile.get('preferred_artist', ''),
                                 placeholder="e.g., Taylor Swift, Arijit Singh")
        
        # Voice input section
        st.markdown("### 🎤 Voice Commands")
        col_voice, col_status = st.columns([1, 2])
        
        with col_voice:
            if st.button("🎤 Use Voice Input", key="voice_btn"):
                with st.spinner("Listening..."):
                    voice_result = st.session_state.voice_handler.listen_for_preferences()
                    if voice_result:
                        st.success(f"Voice input: {voice_result}")
                        # Parse voice input (basic implementation)
                        if " by " in voice_result:
                            parts = voice_result.split(" by ")
                            language = parts[0].strip()
                            artist = parts[1].strip()
                        else:
                            artist = voice_result
        
        with col_status:
            if st.session_state.voice_handler.is_available():
                st.success("🎤 Voice recognition available")
            else:
                st.warning("🎤 Voice recognition not available")
        
        # Save preferences
        if language or artist:
            update_user_preferences(username, {
                'preferred_language': language,
                'preferred_artist': artist
            })
    
    with col2:
        # Current emotion display
        if st.session_state.current_emotion:
            st.markdown(f"""
            <div class="emotion-card">
                <h3>Current Emotion</h3>
                <h2>{st.session_state.current_emotion.upper()}</h2>
                <p>Detected via AI</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 How it works</h3>
                <p>1. Allow camera access</p>
                <p>2. Enter preferences</p>
                <p>3. Let AI analyze your emotions</p>
                <p>4. Get personalized music!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Camera section
    if language and artist:
        st.markdown("### 📹 Emotion Detection Camera")
        st.info("Position yourself in the camera frame for emotion detection")
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        webrtc_streamer(
            key="emotion_detection",
            desired_playing_state=True,
            video_processor_factory=lambda: EmotionProcessor(username),
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Music platform recommendations
    st.markdown("### 🎯 Get Music Recommendations")
    
    if st.session_state.current_emotion and language and artist:
        music_platforms = MusicPlatforms()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔴 YouTube", use_container_width=True):
                music_platforms.open_youtube(language, artist, st.session_state.current_emotion)
                
        with col2:
            if st.button("🎵 YouTube Music", use_container_width=True):
                music_platforms.open_youtube_music(language, artist, st.session_state.current_emotion)
                
        with col3:
            if st.button("🟢 Spotify", use_container_width=True):
                music_platforms.open_spotify(language, artist, st.session_state.current_emotion)
                
        with col4:
            if st.button("🍎 Apple Music", use_container_width=True):
                music_platforms.open_apple_music(language, artist, st.session_state.current_emotion)
        
        # Reset emotion button
        if st.button("🔄 Reset Emotion Detection", use_container_width=True):
            st.session_state.current_emotion = ""
            st.success("Emotion detection reset!")
    else:
        st.warning("Please fill in preferences and ensure emotion is detected to get recommendations")

def show_games_page():
    """Games integration with Poki"""
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Games Hub</h1>
        <p>Play games while waiting for emotion detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    games_integration = GamesIntegration()
    games = games_integration.get_popular_games()
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search Games", placeholder="Search for games...")
    with col2:
        category = st.selectbox("Category", ["All", "Action", "Puzzle", "Racing", "Sports"])
    
    # Display games in grid
    cols = st.columns(3)
    for i, game in enumerate(games):
        if search_query and search_query.lower() not in game['name'].lower():
            continue
        if category != "All" and category.lower() not in game['category'].lower():
            continue
            
        with cols[i % 3]:
            st.markdown(f"""
            <div class="game-card">
                <img src="{game['image']}" width="100%" style="border-radius: 10px;">
                <h4>{game['name']}</h4>
                <p>{game['description']}</p>
                <p><strong>Category:</strong> {game['category']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Play {game['name']}", key=f"game_{i}"):
                games_integration.open_game(game['poki_url'])
                # Track game play
                username = st.session_state.get('username')
                games_integration.track_game_play(username, game['name'])

def show_history_page():
    """Show emotion detection history"""
    st.markdown("""
    <div class="main-header">
        <h1>📜 Emotion History</h1>
        <p>Track your emotional journey over time</p>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.session_state.get('username')
    
    # Date filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("From", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", datetime.now())
    with col3:
        emotion_filter = st.multiselect("Filter Emotions", 
                                       labels if labels is not None else [])
    
    # Get history data
    history = get_emotion_history(username, start_date, end_date, emotion_filter)
    
    if history:
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Detections", len(df))
        with col2:
            st.metric("Unique Emotions", df['emotion'].nunique())
        with col3:
            st.metric("Most Common", df['emotion'].mode().iloc[0] if not df.empty else "None")
        with col4:
            st.metric("Average per Day", f"{len(df) / max(1, (end_date - start_date).days):.1f}")
        
        # Timeline chart
        st.markdown("### 📈 Emotion Timeline")
        timeline_df = df.groupby([df['timestamp'].dt.date, 'emotion']).size().reset_index()
        timeline_df.columns = ['date', 'emotion', 'count']
        
        fig = px.line(timeline_df, x='date', y='count', color='emotion',
                     title="Emotion Detection Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history table
        st.markdown("### 📋 Detailed History")
        st.dataframe(df[['timestamp', 'emotion', 'language', 'artist']].sort_values('timestamp', ascending=False),
                    use_container_width=True)
    else:
        st.info("No emotion history found for the selected period.")

def show_analytics_page():
    """Advanced analytics dashboard"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Understand your emotional patterns and music preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.session_state.get('username')
    analytics = EmotionAnalytics(username)
    
    # Time range selector
    time_range = st.selectbox("Select Time Range", 
                             ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
    
    # Get analytics data
    analytics_data = analytics.get_comprehensive_analytics(time_range)
    
    if analytics_data:
        # Emotion frequency pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Emotion Distribution")
            emotion_counts = analytics_data['emotion_frequency']
            fig = px.pie(values=list(emotion_counts.values()), 
                        names=list(emotion_counts.keys()),
                        title="Emotion Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📅 Daily Pattern")
            daily_pattern = analytics_data['daily_pattern']
            fig = px.bar(x=list(daily_pattern.keys()), 
                        y=list(daily_pattern.values()),
                        title="Detections by Hour of Day")
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekly pattern
        st.markdown("### 📆 Weekly Pattern")
        weekly_pattern = analytics_data['weekly_pattern']
        fig = px.bar(x=list(weekly_pattern.keys()), 
                    y=list(weekly_pattern.values()),
                    title="Detections by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mood trends
        st.markdown("### 📈 Mood Trends")
        mood_trends = analytics_data['mood_trends']
        if mood_trends:
            df_trends = pd.DataFrame(mood_trends)
            fig = px.line(df_trends, x='date', y='mood_score', 
                         title="Mood Score Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights and recommendations
        st.markdown("### 💡 Insights & Recommendations")
        insights = analytics.get_insights(analytics_data)
        
        for insight in insights:
            st.info(f"💡 {insight}")
    else:
        st.info("Not enough data for analytics. Start using the emotion detection to build your analytics!")

def show_profile_page():
    """User profile management"""
    st.markdown("""
    <div class="main-header">
        <h1>👤 My Profile</h1>
        <p>Manage your account and preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.session_state.get('username')
    user_profile = get_user_profile(username)
    
    if user_profile:
        # Profile information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Profile Information")
            
            # Editable profile fields
            with st.form("profile_form"):
                email = st.text_input("Email", value=user_profile.get('email', ''))
                phone = st.text_input("Phone", value=user_profile.get('phone', ''))
                preferred_language = st.text_input("Preferred Language", 
                                                 value=user_profile.get('preferred_language', ''))
                preferred_artist = st.text_input("Preferred Artist", 
                                                value=user_profile.get('preferred_artist', ''))
                
                # Music platform preferences
                st.markdown("#### 🎵 Music Platform Preferences")
                spotify_connected = st.checkbox("Connect Spotify", 
                                               value=user_profile.get('spotify_connected', False))
                youtube_music_connected = st.checkbox("Connect YouTube Music", 
                                                    value=user_profile.get('youtube_music_connected', False))
                
                # Notification preferences
                st.markdown("#### 🔔 Notification Preferences")
                email_notifications = st.checkbox("Email Notifications", 
                                                 value=user_profile.get('email_notifications', True))
                emotion_alerts = st.checkbox("Emotion Pattern Alerts", 
                                           value=user_profile.get('emotion_alerts', False))
                
                if st.form_submit_button("💾 Save Profile"):
                    update_data = {
                        'email': email,
                        'phone': phone,
                        'preferred_language': preferred_language,
                        'preferred_artist': preferred_artist,
                        'spotify_connected': spotify_connected,
                        'youtube_music_connected': youtube_music_connected,
                        'email_notifications': email_notifications,
                        'emotion_alerts': emotion_alerts
                    }
                    
                    if update_user_preferences(username, update_data):
                        st.success("Profile updated successfully!")
                    else:
                        st.error("Failed to update profile")
        
        with col2:
            st.markdown("### 📊 Account Statistics")
            
            analytics = EmotionAnalytics(username)
            stats = analytics.get_comprehensive_stats()
            
            # Display stats cards
            stats_to_show = [
                ("Total Sessions", stats.get('total_sessions', 0), "🎵"),
                ("Emotions Detected", stats.get('total_emotions', 0), "😊"),
                ("Games Played", stats.get('games_played', 0), "🎮"),
                ("Days Active", stats.get('days_active', 0), "📅"),
            ]
            
            for stat_name, stat_value, icon in stats_to_show:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>{icon} {stat_name}</h3>
                    <h2>{stat_value}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Account actions
            st.markdown("### ⚙️ Account Actions")
            
            if st.button("📊 Export Data", use_container_width=True):
                # Export user data
                export_data = analytics.export_user_data()
                st.download_button(
                    label="📥 Download Data",
                    data=export_data,
                    file_name=f"{username}_data_export.json",
                    mime="application/json"
                )
            
            if st.button("🗑️ Delete Account", use_container_width=True):
                st.error("This action cannot be undone!")
                confirm = st.checkbox("I understand this will delete all my data")
                if confirm and st.button("❌ Confirm Deletion"):
                    # Implement account deletion
                    st.error("Account deletion feature coming soon!")
    
    else:
        st.error("Profile not found. Please contact support.")

def main():
    """Main application entry point"""
    # Check authentication
    if not is_authenticated():
        show_auth_page()
        st.stop()
    
    # Load custom CSS
    load_css()
    
    # Show navigation and user info
    show_navigation()
    show_user_info()
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == "Player":
        show_player_page()
    elif current_page == "Games":
        show_games_page()
    elif current_page == "History":
        show_history_page()
    elif current_page == "Analytics":
        show_analytics_page()
    elif current_page == "Profile":
        show_profile_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎵 Enhanced AI Music Player - Your emotions, your music</p>
        <p>Built with ❤️ using Streamlit, MediaPipe, TensorFlow, and AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()