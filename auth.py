import streamlit as st
import pymongo
import hashlib
import re
import jwt
from datetime import datetime, timedelta
import os
from streamlit_oauth import OAuth2Component
import secrets
import bcrypt
import time

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")

class AuthenticationSystem:
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(MONGODB_URI)
            self.db = self.client['enhanced_music_app']
            self.users_collection = self.db['users']
            
            # Create indexes for better performance
            self.users_collection.create_index("username", unique=True)
            self.users_collection.create_index("email", unique=True)
            
        except Exception as e:
            st.error(f"Database connection failed: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        if isinstance(hashed, str):
            hashed = hashed.encode('utf-8')
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        pattern = r'^[+]?[1-9]?[0-9]{7,15}$'
        return re.match(pattern, phone) is not None
    
    def validate_password(self, password: str) -> tuple:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        return True, "Password is strong"
    
    def register_user(self, username: str, email: str, phone: str, password: str, 
                     confirm_password: str) -> tuple:
        """Register a new user"""
        try:
            # Validation
            if not all([username, email, phone, password, confirm_password]):
                return False, "All fields are required"
            
            if password != confirm_password:
                return False, "Passwords do not match"
            
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            if not self.validate_phone(phone):
                return False, "Invalid phone number format"
            
            password_valid, password_message = self.validate_password(password)
            if not password_valid:
                return False, password_message
            
            # Check if user exists
            existing_user = self.users_collection.find_one({
                "$or": [
                    {"username": username},
                    {"email": email}
                ]
            })
            
            if existing_user:
                return False, "Username or email already exists"
            
            # Create user document
            hashed_password = self.hash_password(password)
            user_doc = {
                "username": username,
                "email": email,
                "phone": phone,
                "password": hashed_password,
                "created_at": datetime.utcnow(),
                "last_login": None,
                "login_method": "email",
                "email_verified": False,
                "profile": {
                    "preferred_language": "",
                    "preferred_artist": "",
                    "spotify_connected": False,
                    "youtube_music_connected": False,
                    "email_notifications": True,
                    "emotion_alerts": False
                },
                "stats": {
                    "total_sessions": 0,
                    "total_emotions": 0,
                    "games_played": 0,
                    "days_active": 0
                }
            }
            
            # Insert user
            result = self.users_collection.insert_one(user_doc)
            if result.inserted_id:
                return True, "Registration successful! Please login."
            else:
                return False, "Registration failed. Please try again."
                
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def login_user(self, username: str, password: str) -> tuple:
        """Login user with username/email and password"""
        try:
            # Find user by username or email
            user = self.users_collection.find_one({
                "$or": [
                    {"username": username},
                    {"email": username}
                ]
            })
            
            if not user:
                return False, "Invalid credentials"
            
            # Verify password
            if not self.verify_password(password, user['password']):
                return False, "Invalid credentials"
            
            # Update last login
            self.users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {"last_login": datetime.utcnow()},
                    "$inc": {"stats.total_sessions": 1}
                }
            )
            
            # Create session
            self._create_session(user['username'], user['email'])
            
            return True, "Login successful!"
            
        except Exception as e:
            return False, f"Login error: {str(e)}"
    
    def google_oauth_login(self):
        """Handle Google OAuth login"""
        try:
            oauth2 = OAuth2Component(
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
            )
            
            result = oauth2.authorize_button(
                name="Continue with Google",
                icon="https://developers.google.com/identity/images/g-logo.png",
                redirect_uri="http://localhost:8501",
                scope="openid email profile",
                key="google_oauth",
                use_container_width=True
            )
            
            if result and "token" in result:
                # Get user info from Google
                user_info = result.get("user", {})
                email = user_info.get("email", "")
                name = user_info.get("name", "")
                
                if email:
                    # Check if user exists
                    existing_user = self.users_collection.find_one({"email": email})
                    
                    if existing_user:
                        # Update last login
                        self.users_collection.update_one(
                            {"email": email},
                            {
                                "$set": {"last_login": datetime.utcnow()},
                                "$inc": {"stats.total_sessions": 1}
                            }
                        )
                        username = existing_user['username']
                    else:
                        # Create new user
                        username = email.split('@')[0]
                        # Ensure username is unique
                        counter = 1
                        original_username = username
                        while self.users_collection.find_one({"username": username}):
                            username = f"{original_username}{counter}"
                            counter += 1
                        
                        user_doc = {
                            "username": username,
                            "email": email,
                            "phone": "",
                            "password": "",  # No password for OAuth users
                            "created_at": datetime.utcnow(),
                            "last_login": datetime.utcnow(),
                            "login_method": "google",
                            "email_verified": True,
                            "profile": {
                                "preferred_language": "",
                                "preferred_artist": "",
                                "spotify_connected": False,
                                "youtube_music_connected": False,
                                "email_notifications": True,
                                "emotion_alerts": False
                            },
                            "stats": {
                                "total_sessions": 1,
                                "total_emotions": 0,
                                "games_played": 0,
                                "days_active": 1
                            }
                        }
                        
                        self.users_collection.insert_one(user_doc)
                    
                    # Create session
                    self._create_session(username, email)
                    
                    return True, "Google login successful!"
                else:
                    return False, "Failed to get user information from Google"
            
            return False, "Google authentication cancelled"
            
        except Exception as e:
            return False, f"Google OAuth error: {str(e)}"
    
    def _create_session(self, username: str, email: str):
        """Create user session"""
        session_data = {
            'username': username,
            'email': email,
            'login_time': datetime.utcnow().isoformat(),
            'exp': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        
        # Create JWT token
        token = jwt.encode(session_data, JWT_SECRET, algorithm='HS256')
        
        # Store in session state
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['email'] = email
        st.session_state['auth_token'] = token
    
    def logout(self):
        """Logout user and clear session"""
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key.startswith(('logged_in', 'username', 'email', 'auth_token', 'current_')):
                del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated (relaxed for troubleshooting)"""
        return st.session_state.get('logged_in', False)
    
    def get_current_user(self) -> dict:
        """Get current user information"""
        username = st.session_state.get('username')
        if username:
            return self.users_collection.find_one({"username": username})
        return None
    
    def reset_password_request(self, email: str) -> tuple:
        """Request password reset (placeholder)"""
        # In a real implementation, you would send an email with reset link
        user = self.users_collection.find_one({"email": email})
        if user:
            # Generate reset token and save to database
            reset_token = secrets.token_urlsafe(32)
            expiry = datetime.utcnow() + timedelta(hours=24)
            
            self.users_collection.update_one(
                {"email": email},
                {
                    "$set": {
                        "reset_token": reset_token,
                        "reset_token_expiry": expiry
                    }
                }
            )
            
            return True, "Password reset email sent (feature not fully implemented)"
        else:
            return False, "Email not found"

# Initialize authentication system
auth_system = AuthenticationSystem()

# Wrapper functions for compatibility
def is_authenticated():
    return auth_system.is_authenticated()

def logout():
    auth_system.logout()
    st.rerun()

def show_auth_page():
    """Show authentication page with enhanced UI"""
    # st.set_page_config(
    #     page_title="Enhanced Music Player - Login",
    #     page_icon="🎵",
    #     layout="centered",
    #     initial_sidebar_state="collapsed"
    # )
    
    # Custom CSS for auth page
    st.markdown("""
    <style>
    .auth-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        background: white;
        border: 1px solid #e0e0e0;
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .auth-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .auth-toggle {
        display: flex;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .auth-toggle button {
        flex: 1;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
        background: transparent;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .auth-toggle button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .google-btn {
        width: 100%;
        padding: 0.75rem;
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .google-btn:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    
    .divider {
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
    }
    
    .divider::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: #e0e0e0;
    }
    
    .divider span {
        background: white;
        padding: 0 1rem;
        color: #666;
    }
    
    .form-section {
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .password-strength {
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.875rem;
    }
    
    .strength-weak { background: #ffe6e6; color: #d63031; }
    .strength-medium { background: #fff3cd; color: #856404; }
    .strength-strong { background: #d4edda; color: #155724; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    # Main container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="auth-header">
        <h1>🎵 Music Player</h1>
        <p>Discover music that matches your emotions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle between login and register
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign In", key="login_tab", 
                    use_container_width=True):
            st.session_state.show_register = False
            st.rerun()
    with col2:
        if st.button("Sign Up", key="register_tab", 
                    use_container_width=True):
            st.session_state.show_register = True
            st.rerun()
    
    # Google OAuth
    st.markdown('<div class="divider"><span>Or continue with</span></div>', 
               unsafe_allow_html=True)
    
    # Google login button
    if st.button("🔍 Continue with Google", key="google_login", 
                use_container_width=True):
        success, message = auth_system.google_oauth_login()
        if success:
            st.success(message)
            time.sleep(1)
            st.rerun()
        else:
            st.error(message)
    
    st.markdown('<div class="divider"><span>Or use email</span></div>', 
               unsafe_allow_html=True)
    
    if not st.session_state.show_register:
        # Login Form
        st.markdown("### Sign In to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email", 
                                   placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", 
                                   placeholder="Enter your password")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                login_btn = st.form_submit_button("🔐 Sign In", 
                                                use_container_width=True)
            with col2:
                forgot_btn = st.form_submit_button("Forgot?", 
                                                 use_container_width=True)
            
            if login_btn and username and password:
                success, message = auth_system.login_user(username, password)
                if success:
                    auth_system._create_session(username, "")
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
            
            if forgot_btn and username:
                if auth_system.validate_email(username):
                    success, message = auth_system.reset_password_request(username)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.error("Please enter a valid email address")
    
    else:
        # Register Form
        st.markdown("### Create New Account")
        
        with st.form("register_form"):
            username = st.text_input("Username", 
                                   placeholder="Choose a unique username")
            email = st.text_input("Email", 
                                placeholder="Enter your email address")
            phone = st.text_input("Phone Number", 
                                placeholder="Enter your phone number")
            password = st.text_input("Password", type="password", 
                                   placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", 
                                           placeholder="Confirm your password")
            
            # Real-time password strength indicator
            if password:
                is_valid, strength_message = auth_system.validate_password(password)
                if is_valid:
                    st.success(f"✅ {strength_message}")
                else:
                    st.error(f"❌ {strength_message}")
            
            # Terms and conditions
            terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            register_btn = st.form_submit_button("🎵 Create Account", 
                                               use_container_width=True)
            
            if register_btn:
                if not terms_accepted:
                    st.error("Please accept the Terms of Service and Privacy Policy")
                else:
                    success, message = auth_system.register_user(
                        username, email, phone, password, confirm_password
                    )
                    if success:
                        st.success(message)
                        st.balloons()
                        st.session_state.show_register = False
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.875rem;'>
        <p>🎵 Enhanced AI Music Player</p>
        <p>Built with ❤️ for music lovers</p>
    </div>
    """, unsafe_allow_html=True)

# For backward compatibility
def register_user(*args, **kwargs):
    return auth_system.register_user(*args, **kwargs)

def login_user(*args, **kwargs):
    return auth_system.login_user(*args, **kwargs)