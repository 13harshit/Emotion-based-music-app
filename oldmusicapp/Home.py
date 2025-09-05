import streamlit as st
import pymongo
import random

st.title("Login / Sign Up Page")

# --- Database Connection (MongoDB) ---
def get_db_connection():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["music_app"]
    return db["users"]

# --- Sign Up ---
def send_otp(email, otp):
    # Placeholder for sending OTP via email
    pass

def signup():
    st.subheader("Sign Up")
    username = st.text_input("Username", key="su_username")
    mobile = st.text_input("Mobile Number", key="su_mobile")
    email = st.text_input("Email", key="su_email")
    password = st.text_input("Password", type="password", key="su_password")
    if st.button("Send OTP"):
        otp = str(random.randint(100000, 999999))
        st.session_state["otp"] = otp
        send_otp(email, otp)
        st.success(f"OTP sent to {email} (demo: {otp})")
    otp_input = st.text_input("Enter OTP", key="su_otp")
    if st.button("Sign Up"):
        if otp_input == st.session_state.get("otp", ""):
            users = get_db_connection()
            if users.find_one({"username": username}):
                st.error("Username already exists.")
            else:
                users.insert_one({
                    "username": username,
                    "mobile": mobile,
                    "email": email,
                    "password": password
                })
                st.success("Sign up successful! Please sign in.")
        else:
            st.error("Invalid OTP.")

# --- Sign In ---
def signin():
    st.subheader("Sign In")
    username = st.text_input("Username", key="si_username")
    password = st.text_input("Password", type="password", key="si_password")
    if st.button("Login"):
        users = get_db_connection()
        user = users.find_one({"username": username, "password": password})
        if user:
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()  # <-- use this instead of st.experimental_rerun()
        else:
            st.error("Invalid credentials.")

# --- Google Sign-In (Demo Link) ---
def google_signin():
    st.subheader("Or Sign In with Google")
    st.markdown("[Google Sign-In (Demo)](https://accounts.google.com/signin)")

# --- Main ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

menu = st.sidebar.selectbox("Menu", ["Sign In", "Sign Up", "Google Sign-In"])

if menu == "Sign In":
    signin()
elif menu == "Sign Up":
    signup()
elif menu == "Google Sign-In":
    google_signin()

if st.session_state["logged_in"]:
    st.success("Login successful! Please select 'Music' from the sidebar.")