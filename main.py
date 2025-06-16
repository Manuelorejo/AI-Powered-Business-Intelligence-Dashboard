#Imports

import streamlit as st
import pandas as pd
import pdfplumber
import time
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import torch
from groq import Groq
from docx import Document
import plotly.express as px
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sqlalchemy import create_engine
import hashlib
import sqlite3

torch.set_default_tensor_type('torch.FloatTensor')

st.set_page_config(layout='wide')



api_key = os.getenv("api_key")


# Database engine
engine = create_engine("sqlite:///data_storage.db")
@st.cache_resource
def load_encoder():
     # Set tensor type first
    torch.set_default_tensor_type('torch.FloatTensor')
    # Load model with explicit device mapping
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    # Explicitly move to CPU if needed
    if any(t.is_meta for t in model.parameters()):
        model = model.to_empty('cpu')
    else:
        model = model.to_empty('cpu')
    return model

model = load_encoder()

# ------------------ Database Setup ------------------
def init_database():
    """Initialize database tables for users and chat history"""
    with sqlite3.connect("data_storage.db") as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                file_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                file_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()

# ------------------ Authentication Functions ------------------
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def register_user(username, email, password):
    """Register a new user"""
    try:
        with sqlite3.connect("data_storage.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, hash_password(password)))
            conn.commit()
            return True, "Registration successful!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists!"
        elif "email" in str(e):
            return False, "Email already exists!"
        else:
            return False, "Registration failed!"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user login"""
    try:
        with sqlite3.connect("data_storage.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash FROM users 
                WHERE username = ?
            ''', (username,))
            user = cursor.fetchone()
            
            if user and verify_password(password, user[2]):
                return True, {"id": user[0], "username": user[1]}
            else:
                return False, "Invalid username or password!"
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

def save_chat_message(user_id, session_id, question, answer, file_name):
    """Save chat message to database"""
    try:
        with sqlite3.connect("data_storage.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (user_id, session_id, question, answer, file_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_id, question, answer, file_name))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Failed to save chat: {str(e)}")
        return False

def get_user_chat_history(user_id, session_id=None):
    """Get chat history for a user"""
    try:
        with sqlite3.connect("data_storage.db") as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute('''
                    SELECT question, answer, file_name, timestamp FROM chat_history 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY timestamp ASC
                ''', (user_id, session_id))
            else:
                cursor.execute('''
                    SELECT question, answer, file_name, timestamp, session_id FROM chat_history 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                ''', (user_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Failed to get chat history: {str(e)}")
        return []

def get_user_sessions(user_id):
    """Get all sessions for a user"""
    try:
        with sqlite3.connect("data_storage.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT session_id, file_name, MIN(timestamp) as created_at
                FROM chat_history 
                WHERE user_id = ?
                GROUP BY session_id, file_name
                ORDER BY created_at DESC
            ''', (user_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Failed to get sessions: {str(e)}")
        return []

# ------------------ Authentication UI ------------------
def show_auth_page():
    """Show login/registration page"""
    st.title("🔐 AI Business Intelligence Dashboard")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            
            if login_btn:
                if username and password:
                    success, result = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = result
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            register_btn = st.form_submit_button("Register")
            
            if register_btn:
                if new_username and new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            success, message = register_user(new_username, new_email, new_password)
                            if success:
                                st.success(message)
                                st.info("Please login with your new account")
                            else:
                                st.error(message)
                        else:
                            st.error("Password must be at least 6 characters long")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

# ------------------ Utility Functions ------------------
def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def ydata(df):
    with st.spinner("Generating Report..."):
        profile = ProfileReport(df, title="Data Profile", explorative=True)
        components.html(profile.to_html(), height=1500, width=1500, scrolling=True)

def visualization(df, column):
    if df[column].dtype == 'object':
        st.bar_chart(df[column].value_counts())
    else:
        st.plotly_chart(px.histogram(df, x=column))

def store_vectors(df, prompt):
    rows_as_text = df.astype(str).apply(lambda row: ("|").join(row), axis=1).tolist()
    embeddings = model.encode(rows_as_text, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, "table_index.faiss")
    with open("table_metadata", "w", encoding="utf-8") as f:
        for row in rows_as_text:
            f.write(row + "\n")
            
    query = prompt
    query_embedding = model.encode([query])
    
    D, I = index.search(np.array(query_embedding), k=5)
    
    with open("table_metadata", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    top_matches = [lines[i] for i in I[0]]
    return top_matches

def chat_with_file(user_question: str, df: pd.DataFrame):
    # Convert a preview of the data to JSON (limit to 50 rows)
    data = df.head(50).to_dict(orient="records")
    data_json = json.dumps(data, indent=2)

    # Construct the prompt for the LLM
    prompt = f"""
You are an AI data analyst. A user has uploaded a dataset and would like to understand it better.
Here is a preview of the dataset (first 50 rows):

{data_json}

Now answer the user's question, based on this data.

User's Question: {user_question}

Answer:
"""

    # Connect to Groq API
    client = Groq(api_key=api_key)

    # Run chat completion
    completion = client.chat.completions.create(
        model="meta-llama/llama-2-7b-chat-hf",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=True
    )

    # Stream the response in Streamlit
    response_text = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        response_text += content

    return response_text

# ------------------ Main Application ------------------
def main_app():
    """Main application after authentication"""
    
    # User info in sidebar
    with st.sidebar:
        st.write(f"👋 Welcome, **{st.session_state.user['username']}**!")
        
        # Chat History Section
        st.subheader("📚 Your Chat Sessions")
        sessions = get_user_sessions(st.session_state.user['id'])
        
        if sessions:
            session_options = [f"{session[1]} ({session[0][:8]}...)" for session in sessions]
            selected_session_idx = st.selectbox("Select a session:", range(len(sessions)), 
                                               format_func=lambda x: session_options[x])
            
            if st.button("Load Session"):
                selected_session = sessions[selected_session_idx]
                st.session_state.current_session = selected_session[0]
                st.session_state.show_history = True
        
        if st.button("View All Chat History"):
            st.session_state.show_all_history = True
            
        if st.button("Logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.title("AI-POWERED BUSINESS INTELLIGENCE DASHBOARD")
    
    # Show chat history if requested
    if st.session_state.get('show_all_history', False):
        st.subheader("📖 Your Complete Chat History")
        history = get_user_chat_history(st.session_state.user['id'])
        
        if history:
            for chat in history:
                with st.expander(f"Q: {chat[0][:50]}... | File: {chat[2]} | {chat[3]}"):
                    st.write(f"**Question:** {chat[0]}")
                    st.write(f"**Answer:** {chat[1]}")
                    st.write(f"**File:** {chat[2]}")
                    st.write(f"**Time:** {chat[3]}")
        else:
            st.info("No chat history found.")
        
        if st.button("Back to Dashboard"):
            st.session_state.show_all_history = False
            st.rerun()
        return
    
    # Show specific session history if requested
    if st.session_state.get('show_history', False):
        st.subheader(f"📖 Session: {st.session_state.current_session}")
        history = get_user_chat_history(st.session_state.user['id'], st.session_state.current_session)
        
        if history:
            for chat in history:
                with st.expander(f"Q: {chat[0][:50]}... | {chat[3]}"):
                    st.write(f"**Question:** {chat[0]}")
                    st.write(f"**Answer:** {chat[1]}")
                    st.write(f"**File:** {chat[2]}")
                    st.write(f"**Time:** {chat[3]}")
        
        if st.button("Back to Dashboard"):
            st.session_state.show_history = False
            st.rerun()
        return

    # File Upload
    uploaded_file = st.file_uploader("Please Upload Your File Here", type=["csv", "pdf", "docx", "xlsx"])

    if uploaded_file is not None:
        extension = uploaded_file.name.split(".")[-1].lower()

        if extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif extension == "pdf":
            extracted_text = extract_pdf_text(uploaded_file)
            st.subheader("Extracted PDF Text")
            st.text_area("Contents", extracted_text, height=400)
        elif extension == "docx":
            doc = Document(uploaded_file)
            extracted_text = '\n'.join([para.text for para in doc.paragraphs])
            st.subheader("Extracted Word Document Text")
            st.text_area("Contents", extracted_text, height=400)
        else:
            st.error("Unsupported file type.")
            df = None

        # Report/Visualization for data files
        if 'df' in locals() and isinstance(df, pd.DataFrame):
            
            
            if st.session_state.get("auto_generate", False):
                ydata(df)
                st.session_state.auto_generate = False  # reset flag after displaying report
            # Initialize session state
            if "chatbot" not in st.session_state:
                st.session_state.chatbot = False
            
            # Generate unique session ID for this file interaction
            if "current_session_id" not in st.session_state:
                st.session_state.current_session_id = f"session_{int(time.time())}"
            
            basename = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_").lower()
            timestamp = str(int(time.time()))
            table_name = f"{basename}_{timestamp}"
            
            try:
                df.to_sql(table_name, engine, if_exists='replace', index=False)
            except Exception as e:
                st.write(f"Unable to commit to database {e}")
            
            with st.sidebar:
                def trigger_auto_generate():
                    st.session_state.auto_generate = True
                    st.session_state.chatbot = False  # Turn off chatbot when switching
                
                def trigger_chatbot():
                    st.session_state.chatbot = True
                    st.session_state.auto_generate = False  # Turn off auto report when switching
                
                with st.sidebar:
                    st.button("Auto-Generate Report", on_click=trigger_auto_generate)
                    st.button("Chat With Your File", on_click=trigger_chatbot)
            
            if st.session_state.chatbot:
                with st.expander("💬 Ask the AI About Your Data", expanded=True):
                    user_question = st.text_input("Enter your question")
                    
                    if st.button("ASK AI") and user_question:
                        with st.spinner("Generating response..."):
                            prompt = store_vectors(df, user_question)
                            response = chat_with_file(user_question, df)
                            
                            # Display response
                            st.markdown("### 🤖 AI Answer")
                            st.write(response)
                            
                            # Save to chat history
                            save_chat_message(
                                st.session_state.user['id'],
                                st.session_state.current_session_id,
                                user_question,
                                response,
                                uploaded_file.name
                            )
                            
                            st.success("Response saved to your chat history!")
            
           

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Show appropriate page based on authentication status
    if st.session_state.authenticated:
        main_app()
    else:
        show_auth_page()
