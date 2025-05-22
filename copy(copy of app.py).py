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
st.set_page_config(page_title="CitizenAI", page_icon="🏛️", layout="wide")

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
    
    # Add default admin if doesn't exist
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_id = str(uuid.uuid4())
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute("INSERT INTO users (id, username, password, email, user_type) VALUES (?, ?, ?, ?, ?)",
                 (admin_id, 'admin', password_hash, 'admin@citizenai.gov', 'admin'))
        c.execute("INSERT INTO government_officials (user_id, department, position, approved) VALUES (?, ?, ?, ?)",
                 (admin_id, 'Administration', 'System Administrator', 1))
    
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
            st.title("🏛️ CitizenAI")
            st.subheader("Intelligent Citizen Engagement")
            
            user_options = ["Chat Assistant", "Analytics Dashboard", "About System"]
            
            # Add admin options for government officials and admins
            if user_type in ["government", "admin"]:
                user_options.append("Admin Panel")
            
            selected_page = st.radio("Navigation", user_options)
            
            if selected_page == "Chat Assistant":
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
                st.rerun()  # Add rerun here to prevent attempting to access deleted session state
            
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
        render_chat_interface(tokenizer, model, device, conversation_manager)
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
        st.title("🏛️ CitizenAI Login")
        
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
    
    tabs = st.tabs(["Pending Approvals", "User Management", "System Settings"])
    
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

# Chat interface page (from your original code)
def render_chat_interface(tokenizer, model, device, conversation_manager):
    st.header("🤖 Citizen Assistant")
    
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
            with st.chat_message("assistant", avatar="🏛️"):
                st.write(f"*Query categorized as: {query_category} (Priority: {priority})*")
                st.write(assistant_reply)
            
            # Add to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
            
            # Update conversation context in the manager
            conversation_manager.update_context(st.session_state.conversation_history)

# Dashboard page (from your original code)
def render_dashboard():
    st.header("📊 Citizen Feedback Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Citizen Sentiment Trends")
        sentiment_chart = create_sentiment_chart(st.session_state.sentiment_data)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - 73% positive sentiment in the last 24 hours
        - 15% improvement in citizen satisfaction this week
        - Trending topics: Road repairs, Park maintenance
        """)
    
    with col2:
        st.subheader("Issue Distribution")
        issue_chart = create_issue_distribution_chart(st.session_state.issue_data)
        st.plotly_chart(issue_chart, use_container_width=True)
        
        st.markdown("""
        **Top Concerns:**
        - Infrastructure issues represent 35% of all queries
        - Public services queries have increased by 10% this month
        - Environment-related concerns growing fastest
        """)
    
    st.divider()
    
    # Real-time data section
    st.subheader("🔄 Real-time Citizen Engagement")
    
    # Simulated real-time data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Users", "127", "+12%")
    col2.metric("Queries Today", "543", "+8%")
    col3.metric("Avg Response Time", "1.2 min", "-0.3 min")
    col4.metric("Resolution Rate", "92%", "+3%")
    
    # Recent queries table (simulated data)
    st.subheader("Recent Citizen Queries")
    recent_queries = pd.DataFrame({
        'Time': ['10:23 AM', '10:15 AM', '10:02 AM', '9:55 AM', '9:48 AM'],
        'Query': [
            'How to renew my driver\'s license?',
            'When is the next city council meeting?',
            'How do I report a pothole?',
            'Park renovation timeline?',
            'Business permit application status?'
        ],
        'Category': ['Licenses', 'Governance', 'Infrastructure', 'Parks', 'Business'],
        'Status': ['Resolved', 'Resolved', 'In Progress', 'Resolved', 'Transferred to Dept.'],
    })
    st.dataframe(recent_queries, use_container_width=True)

# About page (from your original code)
def render_about():
    st.header("🏛️ About CitizenAI")
    
    st.markdown("""
    ### Intelligent Citizen Engagement Platform
    
    CitizenAI is a cutting-edge platform leveraging IBM Granite AI to provide citizens with
    instant, accurate information about government services, and help officials understand
    community needs through real-time analytics.
    
    **Core Features:**
    - 💬 Contextual, multi-turn conversations about public services
    - 📊 Real-time sentiment analysis and feedback monitoring
    - 🔄 Automated query routing and prioritization
    - 📈 Trend analysis and predictive insights
    
    **Built for IBM Hackathon 2025**
    """)

if __name__ == "__main__":
    main() 