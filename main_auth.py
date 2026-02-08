import hashlib
import json
import os

import streamlit as st

# ---------------------------------------------------------------------------
# Authentication Functions
# ---------------------------------------------------------------------------
USER_DB_FILE = "users_db.json"


def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_user_database():
    """Load user database from JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {}


def save_user_database(db):
    """Save user database to JSON file"""
    with open(USER_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)


def create_user(username: str, password: str) -> bool:
    """Create a new user account"""
    db = load_user_database()
    if username in db:
        return False
    db[username] = {
        "password_hash": hash_password(password),
    }
    save_user_database(db)
    return True


def verify_user(username: str, password: str) -> bool:
    """Verify user credentials"""
    db = load_user_database()
    if username not in db:
        return False
    return db[username]["password_hash"] == hash_password(password)


# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_username" not in st.session_state:
    st.session_state.auth_username = None


# ---------------------------------------------------------------------------
# Authentication UI
# ---------------------------------------------------------------------------
def show_auth_page():
    """Display the authentication page"""
    st.set_page_config(page_title="Anime Recommendation System - Login", layout="wide")

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🎬 Anime RecSys")
        st.markdown("### Welcome! Please log in to continue")
        st.divider()

        auth_tab1, auth_tab2 = st.tabs(["Login", "_"])

        with auth_tab1:
            with st.form("login_form"):
                st.subheader("🔐 Login")
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input(
                    "Password", type="password", key="login_password"
                )
                login_submit = st.form_submit_button(
                    "Login", use_container_width=True, type="primary"
                )

                if login_submit:
                    if not login_username or not login_password:
                        st.error("Please enter both username and password")
                    elif verify_user(login_username, login_password):
                        st.session_state.authenticated = True
                        st.session_state.auth_username = login_username
                        st.success(f"✅ Welcome back, {login_username}!")

                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        # with auth_tab2:
        #    with st.form("signup_form"):
        #        st.subheader("📝 Create Account")
        #        signup_username = st.text_input(
        #            "Choose Username", key="signup_username"
        #        )
        #        signup_password = st.text_input(
        #            "Choose Password", type="password", key="signup_password"
        #        )
        #        signup_password_confirm = st.text_input(
        #            "Confirm Password", type="password", key="signup_password_confirm"
        #        )
        #        signup_submit = st.form_submit_button(
        #            "Create Account", use_container_width=True, type="primary"
        #        )
        #
        #        if signup_submit:
        #            if not signup_username or not signup_password:
        #                st.error("Please fill in all fields")
        #            elif len(signup_password) < 6:
        #                st.error("Password must be at least 6 characters long")
        #            elif signup_password != signup_password_confirm:
        #                st.error("Passwords do not match")
        #            elif create_user(signup_username, signup_password):
        #                st.success("✅ Account created! Please login.")
        #                st.balloons()
        #            else:
        #                st.error("Username already exists")
        #
        st.divider()

        # Feature highlights
        st.markdown("### ✨ Features")
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        with feat_col1:
            st.markdown("**🎯 Personalized**")
            st.caption("Tailored recommendations")
        with feat_col2:
            st.markdown("**🔍 Advanced Filters**")
            st.caption("Multiple search methods")
        with feat_col3:
            st.markdown("**📊 Track History**")
            st.caption("Monitor your progress")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if not st.session_state.authenticated:
    # Show authentication page
    show_auth_page()
    # st.stop()
else:
    # User is authenticated, load the main application
    try:
        # Import and run the main application
        st.switch_page("pages/_recommender.py")
    except ImportError as e:
        st.error(f"Error loading main application: {e}")
        st.error("Make sure main.py is in the same directory as this file.")

        # Provide logout option
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.auth_username = None
            st.rerun()
