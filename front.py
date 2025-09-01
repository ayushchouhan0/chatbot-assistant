import streamlit as st
import requests
import time
from audio_recorder_streamlit import audio_recorder

# ---- Page Config ----
st.set_page_config(
    page_title="AI Assistant",
    page_icon="AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- API Configuration ----
API_BASE = "https://chatbot-backend-m9su.onrender.com"

# ---- Custom CSS for Clean Interface ----
st.markdown("""
<style>
    /* Hide Streamlit branding */
    .stApp > header {visibility: hidden;}
    .stApp > footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Messages */
    .user-message {
        background: #007bff;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0 10px 50px;
        display: inline-block;
        max-width: 75%;
        float: right;
        clear: both;
        word-wrap: break-word;
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 50px 10px 0;
        display: inline-block;
        max-width: 75%;
        float: left;
        clear: both;
        border-left: 4px solid #007bff;
        word-wrap: break-word;
    }
    
    .system-message {
        background: #e3f2fd;
        color: #1976d2;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        border: 1px solid #bbdefb;
    }
    
    /* Upload area */
    .upload-container {
        background: white;
        border: 2px dashed #007bff;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        color: #000000;
    }
    
    .upload-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 16px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007bff, #0056b3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-connected { background: #d4edda; color: #155724; }
    .status-disconnected { background: #f8d7da; color: #721c24; }
    .status-processing { background: #fff3cd; color: #856404; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 25px;
        padding: 8px 16px;
    }
    
    /* Clear both for message containers */
    .message-container::after {
        content: "";
        display: table;
        clear: both;
    }
</style>
""", unsafe_allow_html=True)

# ---- Initialize Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "general_chat_history" not in st.session_state:
    st.session_state.general_chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "document_info" not in st.session_state:
    st.session_state.document_info = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "general"  # "general" or "document"

# ---- Helper Functions ----
def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def upload_document(uploaded_file):
    """Upload document to backend"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.document_info = {
                "filename": data["filename"],
                "doc_count": data["doc_count"]
            }
            st.session_state.document_uploaded = True
            st.session_state.chat_history = []
            st.session_state.current_mode = "document"
            return True, data["message"]
        else:
            return False, response.json().get('detail', 'Upload failed')
    except Exception as e:
        return False, f"Error: {str(e)}"

def send_message(user_input, session_id=None):
    """Send message to chatbot"""
    try:
        payload = {
            "user_input": user_input,
            "session_id": session_id
        }
        response = requests.post(f"{API_BASE}/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()["bot_response"]
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_audio(audio_bytes):
    """Transcribe audio to text"""
    try:
        files = {"audio": ("recording.wav", audio_bytes, "audio/wav")}
        response = requests.post(f"{API_BASE}/transcribe", files=files, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()["transcription"]
        else:
            return False, response.json().get('detail', 'Transcription failed')
    except Exception as e:
        return False, str(e)

def clear_general_chat():
    """Clear general chat history"""
    try:
        response = requests.post(f"{API_BASE}/clear-general-chat", timeout=10)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

# ---- Main Interface ----
def main():
    # Title
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: white; margin-bottom: 10px;">Document Assistant</h1>
        <p style="color: #e0e0e0; font-size: 18px;">Chat with AI or upload documents for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status Check
    api_status = check_api_status()
    if not api_status:
        st.error("Backend API is not running. Please start the FastAPI server first.")
        st.code("python -m uvicorn backend:app --reload --host 127.0.0.1 --port 8000")
        st.stop()
    
    # API Status indicator
    st.markdown("""
    <div style="position: fixed; top: 10px; right: 10px; z-index: 999;">
        <span class="status-badge status-connected">API Connected</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown("---")
    mode = st.radio(
        "Choose Chat Mode:",
        ["General Chat", "Document Chat"],
        horizontal=True,
        help="General Chat: Chat with AI about anything. Document Chat: Upload and chat with specific documents."
    )
    
    current_mode = "general" if mode == "General Chat" else "document"
    st.session_state.current_mode = current_mode
    
    # Main Container
    with st.container():
        if current_mode == "general":
            # General Chat Mode
            st.markdown("""
            <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: #333;">General AI Chat</h3>
                <small style="color: #666;">Ask me anything! I'm here to help with general questions and conversations.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Display general chat history
            if st.session_state.general_chat_history:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for user_msg, bot_msg in st.session_state.general_chat_history:
                    st.markdown(f'<div class="message-container"><div class="user-message">{user_msg}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="message-container"><div class="bot-message">{bot_msg}</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-message">
                    Welcome! Ask me anything you'd like to know or discuss.
                </div>
                """, unsafe_allow_html=True)
                
                # Sample questions for general chat
                st.markdown("**Try these topics:**")
                cols = st.columns(2)
                sample_questions = [
                    "Tell me about artificial intelligence",
                    "What are the benefits of renewable energy?",
                    "Explain quantum computing",
                    "How does machine learning work?"
                ]
                
                for i, question in enumerate(sample_questions):
                    col = cols[i % 2]
                    with col:
                        if st.button(question, key=f"general_sample_{i}", use_container_width=True):
                            with st.spinner("Thinking..."):
                                bot_response = send_message(question, None)
                                st.session_state.general_chat_history.append((question, bot_response))
                            st.rerun()
            
            # Clear general chat button
            if st.session_state.general_chat_history:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Clear General Chat", use_container_width=True):
                        if clear_general_chat():
                            st.session_state.general_chat_history = []
                            st.success("General chat history cleared!")
                            st.rerun()
        
        else:
            # Document Chat Mode
            if not st.session_state.document_uploaded:
                # Document Upload Phase
                st.markdown("""
                <div class="upload-container">
                    <h2>Upload Your PDF Document</h2>
                    <p>Select a PDF file to start document-based analysis and chat</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Choose a PDF file",
                    type=['pdf'],
                    help="Supported format: PDF files only"
                )
                
                if uploaded_file is not None:
                    # Show file info
                    file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
                    st.info(f"**{uploaded_file.name}** ({file_size:.1f} MB)")
                    
                    # Process button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Process Document", type="primary", use_container_width=True):
                            with st.spinner("Processing document... This may take a moment."):
                                progress_bar = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.01)
                                    progress_bar.progress(i + 1)
                                
                                success, message = upload_document(uploaded_file)
                                progress_bar.empty()
                                
                                if success:
                                    st.success(message)
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(message)
            
            else:
                # Document Chat Interface
                # Document header with switch option
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin: 0; color: #333;">Document: {st.session_state.document_info['filename']}</h3>
                        <small style="color: #666;">{st.session_state.document_info['doc_count']} text chunks • {len(st.session_state.chat_history)} messages</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("New Document"):
                        st.session_state.document_uploaded = False
                        st.session_state.session_id = None
                        st.session_state.document_info = None
                        st.session_state.chat_history = []
                        st.rerun()
                
                # Chat History Display
                if st.session_state.chat_history:
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    for user_msg, bot_msg in st.session_state.chat_history:
                        st.markdown(f'<div class="message-container"><div class="user-message">{user_msg}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="message-container"><div class="bot-message">{bot_msg}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Welcome message with sample questions
                    st.markdown("""
                    <div class="system-message">
                        Great! Your document is ready. Ask me anything about it!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sample Questions
                    st.markdown("**Try these questions:**")
                    cols = st.columns(2)
                    sample_questions = [
                        "What is this document about?",
                        "Summarize the main points",
                        "What are the key findings?",
                        "List important recommendations"
                    ]
                    
                    for i, question in enumerate(sample_questions):
                        col = cols[i % 2]
                        with col:
                            if st.button(question, key=f"sample_{i}", use_container_width=True):
                                with st.spinner("Processing..."):
                                    bot_response = send_message(question, st.session_state.session_id)
                                    st.session_state.chat_history.append((question, bot_response))
                                st.rerun()
                
                # Clear document chat button
                if st.session_state.chat_history:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Clear Chat History", use_container_width=True):
                            st.session_state.chat_history = []
                            st.success("Chat history cleared!")
                            st.rerun()
        
        # Input Section (common for both modes)
        st.markdown("---")
        
        # Input tabs
        tab1, tab2 = st.tabs(["Text Input", "Voice Input"])
        
        with tab1:
            # Text input
            col1, col2 = st.columns([5, 1])
            with col1:
                text_input = st.text_input(
                    "Ask a question",
                    placeholder="Type your question here..." if current_mode == "general" else "Ask about your document...",
                    label_visibility="collapsed",
                    key="text_input"
                )
            with col2:
                text_send = st.button("Send", type="primary", use_container_width=True, key="text_send")
            
            if text_input and text_send:
                with st.spinner("Generating response..."):
                    if current_mode == "general":
                        bot_response = send_message(text_input, None)
                        st.session_state.general_chat_history.append((text_input, bot_response))
                    else:
                        if st.session_state.session_id:
                            bot_response = send_message(text_input, st.session_state.session_id)
                            st.session_state.chat_history.append((text_input, bot_response))
                        else:
                            st.error("Please upload a document first!")
                            st.stop()
                st.rerun()
        
        with tab2:
            # Audio input
            st.markdown("**Record your question:**")
            
            # Audio recorder
            audio_bytes = audio_recorder(
                text="Click to record your question",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="1x",
                pause_threshold=2.0,
                key="audio_recorder"
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Transcribe & Send", type="primary", use_container_width=True):
                        with st.spinner("Converting speech to text..."):
                            success, transcription = transcribe_audio(audio_bytes)
                            
                            if success:
                                st.success(f"Transcribed: '{transcription}'")
                                
                                with st.spinner("Generating response..."):
                                    if current_mode == "general":
                                        bot_response = send_message(transcription, None)
                                        st.session_state.general_chat_history.append((f"[Audio] {transcription}", bot_response))
                                    else:
                                        if st.session_state.session_id:
                                            bot_response = send_message(transcription, st.session_state.session_id)
                                            st.session_state.chat_history.append((f"[Audio] {transcription}", bot_response))
                                        else:
                                            st.error("Please upload a document first!")
                                            st.stop()
                                st.rerun()
                            else:
                                st.error(f"Transcription failed: {transcription}")
                
                with col2:
                    if st.button("Discard Recording", use_container_width=True):
                        st.rerun()

# ---- Run App ----
if __name__ == "__main__":

    main()

