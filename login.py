"""
Login Page Module for FLAME FOX
Provides authentication before accessing the main recommendation app
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta

# -------------------------
# Hardcoded Users Database
# (In production, use a real database like Firebase, PostgreSQL, etc.)
# -------------------------
USERS_DB = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "email": "admin@flamefox.com",
        "created_at": "2025-01-01"
    },
    "user": {
        "password_hash": hashlib.sha256("password".encode()).hexdigest(),
        "email": "user@flamefox.com",
        "created_at": "2025-01-01"
    },
    "demo": {
        "password_hash": hashlib.sha256("demo".encode()).hexdigest(),
        "email": "demo@flamefox.com",
        "created_at": "2025-01-01"
    },
    "NXTWAVE": {
        "password_hash": hashlib.sha256("NIAT29".encode()).hexdigest(),
        "email": "nxtwave@flamefox.com",
        "created_at": "2025-11-16"
    }
}

# -------------------------
# Helper Functions
# -------------------------
def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verify a password against its hash"""
    return stored_hash == hash_password(provided_password)

def is_session_valid() -> bool:
    """Check if session is still valid (not expired)"""
    if 'login_time' not in st.session_state:
        return False
    # Session valid for 24 hours
    session_duration = timedelta(hours=24)
    elapsed = datetime.now() - st.session_state['login_time']
    return elapsed < session_duration

# -------------------------
# Login UI
# -------------------------
def show_login_page():
    """Display the login page"""
    st.set_page_config(page_title="FLAME FOX - Login", layout="centered")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🎭 FLAME FOX</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Mood → Meal · Music · Entertainment</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Tab selection
        tab1, tab2 = st.tabs(["Login", "Demo"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            
            if st.button("🔓 Login", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif username not in USERS_DB:
                    st.error("❌ Username not found. Please check and try again.")
                elif not verify_password(USERS_DB[username]["password_hash"], password):
                    st.error("❌ Incorrect password. Please try again.")
                else:
                    # Login successful
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['login_time'] = datetime.now()
                    st.success(f"✅ Welcome back, {username}!")
                    st.rerun()
            
            st.markdown("---")
            st.markdown(
                "<p style='text-align: center; color: #666; font-size: 0.9em;'>"
                "Demo credentials:<br/>"
                "<code>username: demo | password: demo</code>"
                "</p>",
                unsafe_allow_html=True
            )
        
        with tab2:
            st.subheader("Try Demo Mode")
            st.write("Experience FLAME FOX with pre-set demo credentials:")
            st.info("**Username:**/**Password:** [GIVEN IN THE README FILE]")
            
            if st.button("🚀 Enter Demo Mode", use_container_width=True):
                st.session_state['authenticated'] = True
                st.session_state['username'] = "demo"
                st.session_state['login_time'] = datetime.now()
                st.success("✅ Entering demo mode...")
                st.rerun()

# -------------------------
# Sign Up / Register UI (Optional)
# -------------------------
def show_signup_page():
    """Display the signup page (for future implementation)"""
    st.subheader("Create New Account")
    st.info("Sign-up feature coming soon! Contact us for early access.")
    
    if st.button("← Back to Login"):
        st.session_state['show_signup'] = False
        st.rerun()

# -------------------------
# Logout Function
# -------------------------
def logout():
    """Clear session and return to login page"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.session_state['login_time'] = None
    st.session_state['meal_obj'] = None
    st.session_state['music_obj'] = None
    st.session_state['ent_list'] = None
    st.success("✅ Logged out successfully!")
    st.rerun()

# -------------------------
# Check Authentication
# -------------------------
def check_authentication():
    """
    Check if user is authenticated.
    Returns True if authenticated and session is valid, False otherwise.
    """
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated'] or not is_session_valid():
        st.session_state['authenticated'] = False
        return False
    
    return True
