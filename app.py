import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
import os
import warnings
from dotenv import load_dotenv
from utils.conversation import ConversationManager
from utils.sentiment import SentimentAnalyzer
from utils.automation import QueryClassifier
from utils.visualization import create_sentiment_chart, create_issue_distribution_chart
import hashlib
import sqlite3
import uuid
import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Fix for PyTorch compatibility with Streamlit watcher
import sys
class PathFixer:
    def __init__(self, original_module):
        self.original_module = original_module
    
    def __getattr__(self, attr):
        if attr == "__path__":
            return []
        return getattr(self.original_module, attr)

# Apply the fix for torch.classes module
if "torch.classes" in sys.modules:
    sys.modules["torch.classes"] = PathFixer(sys.modules["torch.classes"])

# Page layout
st.set_page_config(page_title="CitizenAI", page_icon=None, layout="wide")

# Load environment variables
load_dotenv()

# Database setup
def init_db():
    conn = sqlite3.connect('citizenai.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        user_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Create a government officials table
    c.execute('''
    CREATE TABLE IF NOT EXISTS government_officials (
        user_id TEXT PRIMARY KEY,
        department TEXT NOT NULL,
        position TEXT NOT NULL,
        approved BOOLEAN DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create a citizens table
    c.execute('''
    CREATE TABLE IF NOT EXISTS citizens (
        user_id TEXT PRIMARY KEY,
        address TEXT,
        city TEXT,
        state TEXT,
        zipcode TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create a sessions table
    c.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # Create an announcements table
    c.execute('''
    CREATE TABLE IF NOT EXISTS announcements (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        author_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        importance TEXT NOT NULL,
        FOREIGN KEY (author_id) REFERENCES users(id)
    )
    ''')
    
    # Add default admin if doesn't exist
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_id = str(uuid.uuid4())
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute("INSERT INTO users (id, username, password, email, user_type) VALUES (?, ?, ?, ?, ?)",
                 (admin_id, 'admin', password_hash, 'admin@citizenai.gov', 'admin'))
        c.execute("INSERT INTO government_officials (user_id, department, position, approved) VALUES (?, ?, ?, ?)",
                 (admin_id, 'Administration', 'System Administrator', 1))
        
        # Add sample announcements
        announcement_id = str(uuid.uuid4())
        c.execute("INSERT INTO announcements (id, title, content, author_id, importance) VALUES (?, ?, ?, ?, ?)",
                 (announcement_id, 
                  'Road Maintenance Schedule Update', 
                  'The downtown road maintenance project will begin next Monday. Please expect delays on Main Street between 5th and 10th Avenue from 8 AM to 4 PM daily for approximately two weeks. Alternative routes are advised during this period.',
                  admin_id, 'High'))
                  
        announcement_id = str(uuid.uuid4())
        c.execute("INSERT INTO announcements (id, title, content, author_id, importance) VALUES (?, ?, ?, ?, ?)",
                 (announcement_id, 
                  'City Council Meeting Notice', 
                  'The next City Council meeting will be held on May 28th at 7 PM in the City Hall Main Chamber. The agenda includes budget review, parks development plan, and public comments on the new transit proposal.',
                  admin_id, 'Medium'))
                  
        announcement_id = str(uuid.uuid4())
        c.execute("INSERT INTO announcements (id, title, content, author_id, importance) VALUES (?, ?, ?, ?, ?)",
                 (announcement_id, 
                  'Water Conservation Advisory', 
                  'Due to the ongoing drought conditions, all residents are asked to conserve water by limiting outdoor watering to twice weekly. The conservation program will remain in effect until further notice.',
                  admin_id, 'High'))
    
    conn.commit()
    return conn

# Load model (optimized implementation)
@st.cache_resource
def load_model():
    try:
        model_name = "ibm-granite/granite-3.3-2b-instruct"
        
        # Load with optimizations for faster inference
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Enable optimization if available
        if hasattr(model, "eval"):
            model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    conn = st.session_state.db_connection
    c = conn.cursor()
    
    # Get user
    c.execute("SELECT id, password, user_type FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    
    if user and user[1] == hash_password(password):
        # Update last login
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                 (datetime.datetime.now(), user[0]))
        
        # Create session
        session_id = str(uuid.uuid4())
        expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
        c.execute("INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
                 (session_id, user[0], expiry))
        conn.commit()
        
        return user[0], user[2], session_id
    
    return None, None, None

def register_user(username, email, password, user_type, additional_info=None):
    conn = st.session_state.db_connection
    c = conn.cursor()
    
    try:
        # Check if username or email already exists
        c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if c.fetchone():
            return False, "Username or email already exists"
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        c.execute("INSERT INTO users (id, username, password, email, user_type) VALUES (?, ?, ?, ?, ?)",
                 (user_id, username, password_hash, email, user_type))
        
        # Add user type specific info
        if user_type == "citizen" and additional_info:
            c.execute("INSERT INTO citizens (user_id, address, city, state, zipcode) VALUES (?, ?, ?, ?, ?)",
                     (user_id, additional_info.get('address', ''), additional_info.get('city', ''), 
                      additional_info.get('state', ''), additional_info.get('zipcode', '')))
        
        elif user_type == "government" and additional_info:
            c.execute("INSERT INTO government_officials (user_id, department, position, approved) VALUES (?, ?, ?, ?)",
                     (user_id, additional_info.get('department', ''), additional_info.get('position', ''), 0))
        
        conn.commit()
        return True, user_id
    
    except Exception as e:
        conn.rollback()
        return False, str(e)

def check_session():
    if "user_id" in st.session_state and "session_id" in st.session_state:
        conn = st.session_state.db_connection
        c = conn.cursor()
        
        # Check if session exists and is valid
        c.execute("""
            SELECT s.user_id, u.user_type 
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_id = ? AND s.user_id = ? AND s.expires_at > ?
        """, (st.session_state.session_id, st.session_state.user_id, datetime.datetime.now()))
        
        result = c.fetchone()
        if result:
            return True, result[1]
    
    return False, None

def logout():
    if "session_id" in st.session_state:
        conn = st.session_state.db_connection
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE session_id = ?", (st.session_state.session_id,))
        conn.commit()
    
    # Clear session state
    for key in ['user_id', 'username', 'user_type', 'session_id']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.current_page = "login"

# Initialize session state variables
def init_session_state():
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = init_db()
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "current_page" not in st.session_state:
        is_logged_in, _ = check_session()
        st.session_state.current_page = "chat" if is_logged_in else "login"
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "query"
    
    if "sentiment_data" not in st.session_state:
        # Sample sentiment data - in production, replace with real data
        st.session_state.sentiment_data = pd.DataFrame({
            'date': pd.date_range(start='2025-05-10', periods=7, freq='D'),
            'positive': [65, 63, 68, 70, 72, 75, 73],
            'neutral': [20, 22, 18, 17, 16, 15, 17],
            'negative': [15, 15, 14, 13, 12, 10, 10]
        })
    
    if "issue_data" not in st.session_state:
        # Sample issue data - in production, replace with real data
        st.session_state.issue_data = {
            'Infrastructure': 35,
            'Public Services': 25,
            'Environment': 20,
            'Safety': 15,
            'Other': 5
        }

# Initialize managers/utilities
@st.cache_resource
def init_utilities():
    conversation_manager = ConversationManager()
    sentiment_analyzer = SentimentAnalyzer()
    query_classifier = QueryClassifier()
    return conversation_manager, sentiment_analyzer, query_classifier

# Main application
def main():
    # Initialize session state first
    init_session_state()
    
    # Then load model and utilities
    # Use conditional loading to improve startup time
    if "tokenizer" not in st.session_state or "model" not in st.session_state or "device" not in st.session_state:
        # Only load the model if we might need it (user is authenticated or going to authenticate)
        is_logged_in, _ = check_session()
        needs_model = is_logged_in or st.session_state.current_page in ["login", "register", "gov_register"]
        
        if needs_model:
            tokenizer, model, device = load_model()
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.device = device
        else:
            tokenizer, model, device = None, None, None
    else:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        device = st.session_state.device
    
    conversation_manager, sentiment_analyzer, query_classifier = init_utilities()
    
    # Check if user is logged in
    is_logged_in, user_type = check_session()
    
    # Handle navigation and authentication
    if not is_logged_in and st.session_state.current_page not in ["login", "register", "gov_register"]:
        st.session_state.current_page = "login"
    
    # Common header and navigation for authenticated users
    if is_logged_in:
        with st.sidebar:
            st.title("CitizenAI")
            st.subheader("Intelligent Citizen Engagement")
            
            user_options = ["Citizen Assistant", "Analytics Dashboard", "About System"]
            
            # Add admin options for government officials and admins
            if user_type in ["government", "admin"]:
                user_options.append("Admin Panel")
            
            selected_page = st.radio("Navigation", user_options)
            
            if selected_page == "Citizen Assistant":
                st.session_state.current_page = "chat"
            elif selected_page == "Analytics Dashboard":
                st.session_state.current_page = "dashboard"
            elif selected_page == "Admin Panel":
                st.session_state.current_page = "admin"
            elif selected_page == "About System":
                st.session_state.current_page = "about"
            
            st.divider()
            if st.button("Logout"):
                logout()
                st.rerun()
            
            # Check if username exists in session state before displaying it
            if "username" in st.session_state:
                st.markdown(f"**Logged in as: {st.session_state.username}**")
            st.markdown("**IBM Hackathon 2025**")
    
    # Display selected page
    if st.session_state.current_page == "login":
        render_login_page()
    elif st.session_state.current_page == "register":
        render_register_page("citizen")
    elif st.session_state.current_page == "gov_register":
        render_register_page("government")
    elif st.session_state.current_page == "chat":
        render_citizen_assistant(tokenizer, model, device, conversation_manager)
    elif st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "admin":
        render_admin_panel()
    else:
        render_about()

# Login page
def render_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("CitizenAI Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Sign In")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login", use_container_width=True):
                    if username and password:
                        user_id, user_type, session_id = authenticate(username, password)
                        
                        if user_id:
                            # Set session state variables
                            st.session_state.user_id = user_id
                            st.session_state.username = username
                            st.session_state.user_type = user_type
                            st.session_state.session_id = session_id
                            st.session_state.current_page = "chat"
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter username and password")
            
        with tab2:
            st.subheader("Create an Account")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Citizen Registration", use_container_width=True):
                    st.session_state.current_page = "register"
                    st.rerun()
            
            with col2:
                if st.button("Government Official", use_container_width=True):
                    st.session_state.current_page = "gov_register"
                    st.rerun()

# Registration page
def render_register_page(user_type):
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if user_type == "citizen":
            st.title("Citizen Registration")
        else:
            st.title("Government Official Registration")
        
        # Common fields
        username = st.text_input("Username*", key=f"{user_type}_username")
        email = st.text_input("Email*", key=f"{user_type}_email")
        password = st.text_input("Password*", type="password", key=f"{user_type}_password")
        confirm_password = st.text_input("Confirm Password*", type="password", key=f"{user_type}_confirm")
        
        additional_info = {}
        
        # User type specific fields
        if user_type == "citizen":
            st.subheader("Personal Information")
            address = st.text_input("Address")
            col1, col2 = st.columns(2)
            with col1:
                city = st.text_input("City")
            with col2:
                state = st.text_input("State")
            zipcode = st.text_input("Zipcode")
            
            additional_info = {
                'address': address,
                'city': city,
                'state': state,
                'zipcode': zipcode
            }
            
        else:  # Government official
            st.subheader("Official Information")
            department = st.selectbox("Department*", [
                "Public Works", "Health Services", "Parks & Recreation", 
                "Transportation", "Finance", "Safety & Emergency", 
                "Environmental Protection", "Education", "Other"
            ])
            position = st.text_input("Position/Title*")
            
            st.info("Government accounts require approval from an administrator before activation.")
            
            additional_info = {
                'department': department,
                'position': position
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Register", use_container_width=True):
                if not username or not email or not password:
                    st.error("Please fill in all required fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, result = register_user(username, email, password, user_type, additional_info)
                    
                    if success:
                        st.success("Registration successful!")
                        st.info("You can now login with your credentials")
                        time.sleep(2)
                        st.session_state.current_page = "login"
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {result}")
        
        with col2:
            if st.button("Back to Login", use_container_width=True):
                st.session_state.current_page = "login"
                st.rerun()

# Admin panel
def render_admin_panel():
    st.title("Admin Panel")
    
    tabs = st.tabs(["Pending Approvals", "User Management", "System Settings", "Announcements"])
    
    with tabs[0]:
        st.subheader("Government Official Approval Requests")
        
        conn = st.session_state.db_connection
        c = conn.cursor()
        
        # Get pending government officials
        c.execute("""
            SELECT u.id, u.username, u.email, u.created_at, g.department, g.position
            FROM users u
            JOIN government_officials g ON u.id = g.user_id
            WHERE g.approved = 0 AND u.user_type = 'government'
        """)
        
        pending_officials = c.fetchall()
        
        if pending_officials:
            for official in pending_officials:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{official[1]}** ({official[2]})")
                        st.write(f"{official[5]} in {official[4]} department")
                        st.write(f"Registered on: {official[3]}")
                    
                    with col2:
                        if st.button("Approve", key=f"approve_{official[0]}"):
                            c.execute("UPDATE government_officials SET approved = 1 WHERE user_id = ?", (official[0],))
                            conn.commit()
                            st.success("User approved")
                            st.rerun()
                    
                    with col3:
                        if st.button("Reject", key=f"reject_{official[0]}"):
                            # In a real application, you might want to notify the user or move to a rejected table
                            c.execute("DELETE FROM government_officials WHERE user_id = ?", (official[0],))
                            c.execute("DELETE FROM users WHERE id = ?", (official[0],))
                            conn.commit()
                            st.success("User rejected")
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No pending approval requests")
    
    with tabs[1]:
        st.subheader("User Management")
        
        # Get all users
        c.execute("""
            SELECT u.id, u.username, u.email, u.user_type, u.created_at, u.last_login
            FROM users u
            ORDER BY u.created_at DESC
        """)
        
        users = c.fetchall()
        
        # Convert to DataFrame for easier display
        users_df = pd.DataFrame(users, columns=["ID", "Username", "Email", "Type", "Created", "Last Login"])
        st.dataframe(users_df, use_container_width=True)
    
    with tabs[2]:
        st.subheader("System Settings")
        
        st.write("Coming soon: System configuration options")
    
    with tabs[3]:
        st.subheader("Manage Announcements")
        
        # Create new announcement
        with st.expander("Create New Announcement"):
            announcement_title = st.text_input("Announcement Title")
            announcement_content = st.text_area("Announcement Content", height=150)
            announcement_importance = st.selectbox("Importance", ["Low", "Medium", "High"])
            
            if st.button("Publish Announcement"):
                if announcement_title and announcement_content:
                    conn = st.session_state.db_connection
                    c = conn.cursor()
                    
                    announcement_id = str(uuid.uuid4())
                    c.execute("INSERT INTO announcements (id, title, content, author_id, importance) VALUES (?, ?, ?, ?, ?)",
                             (announcement_id, announcement_title, announcement_content, st.session_state.user_id, announcement_importance))
                    
                    conn.commit()
                    st.success("Announcement published successfully!")
                else:
                    st.warning("Please fill in both title and content")
        
        # List existing announcements
        st.subheader("Existing Announcements")
        
        conn = st.session_state.db_connection
        c = conn.cursor()
        
        c.execute("""
            SELECT a.id, a.title, a.content, a.created_at, a.importance, u.username
            FROM announcements a
            JOIN users u ON a.author_id = u.id
            ORDER BY a.created_at DESC
        """)
        
        announcements = c.fetchall()
        
        if announcements:
            for announcement in announcements:
                with st.container():
                    cols = st.columns([4, 1, 1])
                    
                    with cols[0]:
                        st.subheader(announcement[1])
                        st.caption(f"Created on {announcement[3]} by {announcement[5]} | Importance: {announcement[4]}")
                        st.write(announcement[2])
                    
                    with cols[1]:
                        if st.button("Edit", key=f"edit_{announcement[0]}"):
                            st.session_state.editing_announcement = announcement[0]
                            st.rerun()
                    
                    with cols[2]:
                        if st.button("Delete", key=f"delete_{announcement[0]}"):
                            c.execute("DELETE FROM announcements WHERE id = ?", (announcement[0],))
                            conn.commit()
                            st.success("Announcement deleted")
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No announcements found")

# Citizen Assistant page with tabs
def render_citizen_assistant(tokenizer, model, device, conversation_manager):
    st.title("Citizen Assistant")
    
    # Tab selection
    query_tab, announcements_tab = st.tabs(["User Query", "Announcements"])
    
    with query_tab:
        # Display conversation history
        for message in st.session_state.conversation_history:
            role = message["role"]
            content = message["content"]
            
            with st.chat_message(role):
                st.write(content)
        
        # User input
        user_input = st.chat_input("Ask a question about city services...")
        
        if user_input:
            # Add user message to chat
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get priority and category (would normally come from QueryClassifier)
            query_category = "Public Services"  # Example category
            priority = "Medium"  # Example priority
            
            # Process with IBM Granite model
            with st.spinner("Processing your request..."):
                # Get the conversation context
                context = conversation_manager.get_context(st.session_state.conversation_history)
                
                # Prepare prompt with context
                prompt = f"<|user|>\nContext: {context}\nCurrent query: {user_input}\n<|assistant|>\n"
                
                # Generate response using your existing model
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the assistant's reply
                assistant_reply = response.split("<|assistant|>")[-1].strip()
                
                # Add system message about query processing (for demonstration)
                with st.chat_message("assistant"):
                    st.write(f"*Query categorized as: {query_category} (Priority: {priority})*")
                    st.write(assistant_reply)
                
                # Add to conversation history
                st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
                
                # Update conversation context in the manager
                conversation_manager.update_context(st.session_state.conversation_history)
    
    with announcements_tab:
        st.header("Public Announcements")
        
        conn = st.session_state.db_connection
        c = conn.cursor()
        
        c.execute("""
            SELECT a.id, a.title, a.content, a.created_at, a.importance, u.username, u.user_type
            FROM announcements a
            JOIN users u ON a.author_id = u.id
            ORDER BY 
                CASE 
                    WHEN a.importance = 'High' THEN 1 
                    WHEN a.importance = 'Medium' THEN 2 
                    ELSE 3 
                END,
                a.created_at DESC
        """)
        
        announcements = c.fetchall()
        
        if announcements:
            for announcement in announcements:
                with st.container():
                    title_col, date_col = st.columns([3, 1])
                    
                    with title_col:
                        if announcement[4] == "High":
                            st.subheader(f"{announcement[1]} 🔴")
                        elif announcement[4] == "Medium":
                            st.subheader(f"{announcement[1]} 🟠")
                        else:
                            st.subheader(announcement[1])
                    
                    with date_col:
                        st.caption(f"Posted: {announcement[3]}")
                    
                    st.write(announcement[2])
                    
                    # Author info
                    department = ""
                    if announcement[6] in ["government", "admin"]:
                        c.execute("""
                            SELECT department, position
                            FROM government_officials
                            WHERE user_id = (SELECT id FROM users WHERE username = ?)
                        """, (announcement[5],))
                        
                        official_info = c.fetchone()
                        if official_info:
                            department = f" ({official_info[0]} Department, {official_info[1]})"
                    
                    st.caption(f"Posted by: {announcement[5]}{department}")
                    st.divider()
        else:
            st.info("No announcements currently available.")

# Dashboard page
def render_dashboard():
    st.header("Citizen Feedback Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Citizen Sentiment Trends")
        sentiment_chart = create_sentiment_chart(st.session_state.sentiment_data)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Display key metrics
        st.subheader("Key Metrics")
        latest = st.session_state.sentiment_data.iloc[-1]
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Positive Sentiment", f"{latest['positive']}%", f"{latest['positive'] - st.session_state.sentiment_data.iloc[-2]['positive']}%")
        with metrics_cols[1]:
            st.metric("Neutral Sentiment", f"{latest['neutral']}%", f"{latest['neutral'] - st.session_state.sentiment_data.iloc[-2]['neutral']}%")
        with metrics_cols[2]:
            st.metric("Negative Sentiment", f"{latest['negative']}%", f"{latest['negative'] - st.session_state.sentiment_data.iloc[-2]['negative']}%")
    
    with col2:
        st.subheader("Issue Distribution")
        issue_chart = create_issue_distribution_chart(st.session_state.issue_data)
        st.plotly_chart(issue_chart, use_container_width=True)
        
        st.subheader("Top Citizen Concerns")
        concerns = [
            "Road maintenance in downtown area",
            "Public transportation frequency",
            "Park facilities maintenance",
            "Water quality concerns",
            "Street lighting in residential areas"
        ]
        
        for i, concern in enumerate(concerns, 1):
            st.write(f"{i}. {concern}")
    
    # Additional insights
    st.header("Insights and Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Trends")
        st.write("""
        Citizen satisfaction has shown a consistent positive trend over the past week, 
        with a 5% increase in positive sentiment. Public services and infrastructure remain
        the top areas of citizen interest, accounting for 60% of all queries.
        """)
        
        st.subheader("Action Items")
        st.write("""
        1. Address recurring road maintenance concerns in the downtown area
        2. Investigate water quality reports in the eastern district
        3. Improve communication around the upcoming public transportation schedule changes
        4. Follow up on park facilities maintenance requests
        """)
    
    with col2:
        st.subheader("Recommended Focus Areas")
        
        # Placeholder data
        focus_data = {
            'area': ['Infrastructure', 'Public Services', 'Environment', 'Safety'],
            'impact': [8, 7, 5, 6],  # Scale of 1-10
            'effort': [6, 4, 3, 7]   # Scale of 1-10
        }
        
        focus_df = pd.DataFrame(focus_data)
        
        st.dataframe(focus_df, use_container_width=True)
        
        st.write("""
        **Analysis**: Based on citizen feedback and current trends, infrastructure
        improvements would have the highest impact on overall satisfaction. 
        Public services offer a good balance of high impact with moderate effort required.
        """)

# About page
def render_about():
    st.title("About CitizenAI")
    
    st.write("""
    ### Intelligent Citizen Engagement Platform

    CitizenAI is a next-generation platform designed to bridge the gap between local government and citizens.
    By leveraging advanced AI technology, we aim to improve communication, streamline service delivery,
    and enhance civic participation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Features")
        st.markdown("""
        - **Natural Language Processing**: Understand and respond to citizen queries in plain language
        - **Smart Query Routing**: Direct questions to the appropriate government departments
        - **Sentiment Analysis**: Track citizen satisfaction and identify emerging concerns
        - **Data-Driven Insights**: Help officials make informed decisions based on citizen feedback
        - **Accessible Interface**: Available 24/7 to address citizen needs
        """)
    
    with col2:
        st.subheader("Technology Stack")
        st.markdown("""
        - **Foundation Model**: IBM Granite 3.3 2B Instruct
        - **Frontend**: Streamlit interactive web application
        - **Analytics**: Custom sentiment analysis and query classification
        - **Security**: End-to-end encryption and user authentication
        - **Database**: SQLite for rapid prototyping (PostgreSQL for production)
        """)
    
    st.subheader("About the Project")
    st.write("""
    CitizenAI was developed as part of the IBM Hackathon 2025, with the goal of showcasing
    how AI can transform public service delivery. The project demonstrates the practical application
    of foundation models in civic technology and e-governance.
    
    This prototype illustrates the potential for AI-powered citizen engagement tools to create more
    responsive and efficient local governments while improving the citizen experience.
    """)
    
    # Team information
    st.subheader("Development Team")
    team_cols = st.columns(4)
    
    with team_cols[0]:
        st.markdown("**Jane Smith**<br>AI Engineer", unsafe_allow_html=True)
    with team_cols[1]:
        st.markdown("**Alex Johnson**<br>Full-stack Developer", unsafe_allow_html=True)
    with team_cols[2]:
        st.markdown("**Sarah Chen**<br>UX/UI Designer", unsafe_allow_html=True)
    with team_cols[3]:
        st.markdown("**Michael Brown**<br>Data Scientist", unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()