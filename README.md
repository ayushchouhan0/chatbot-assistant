# Document Assistant RAG Chatbot

A comprehensive RAG-based chatbot that supports both general AI chat and document-specific conversations with audio input capabilities.

## Features

- **General AI Chat**: Chat with AI about any topic
- **Document Chat**: Upload PDF documents and chat about their content
- **Audio Input**: Record voice messages that are automatically transcribed
- **Session Management**: Separate chat histories for different modes
- **Clean UI**: Modern, responsive interface
- 
## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

1. Fill in your API keys in `.env`:
- **GROQ_API_KEY**: Get from [Groq Console](https://console.groq.com/)
- **HUGGINGFACEHUB_API_TOKEN**: Get from [HuggingFace](https://huggingface.co/settings/tokens)

### 3. Run the Application

#### Terminal 1 - Start Backend Server:
```bash
python -m uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

#### Terminal 2 - Start Frontend:
```bash
streamlit run frontend.py
```

### 4. Access the Application

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

### General Chat Mode
- Select "General Chat" from the mode selector
- Ask any questions about any topic
- Use text or voice input
- Chat history is maintained separately

### Document Chat Mode
- Select "Document Chat" from the mode selector
- Upload a PDF document
- Ask questions about the document content
- The system will retrieve relevant information from the document
- Document chat history is maintained per session

### Audio Input
- Click the microphone button in the "Voice Input" tab
- Record your question
- The audio will be automatically transcribed and sent as text
- Works for both general and document chat modes

## File Structure

```
project/
├── backend.py          # Merged backend server (FastAPI + Core logic)
├── frontend.py         # Streamlit frontend
├── requirements.txt    # Python dependencies
├── .env                # Your actual environment variables (create this)
└── README.md           # This file
```

## API Endpoints

- `POST /upload` - Upload PDF document
- `POST /chat` - Chat (general or document-based)
- `POST /transcribe` - Transcribe audio to text
- `POST /chat-audio` - Direct audio chat
- `GET /health` - Health check
- `DELETE /sessions/{session_id}` - Clear document session
- `POST /clear-general-chat` - Clear general chat history

## Troubleshooting

### Common Issues

1. **API Connection Error**: Make sure the backend server is running on port 8000
2. **Audio Transcription Error**: Ensure your Groq API key is valid and has audio transcription enabled
3. **Document Processing Error**: Check if the PDF is valid and not corrupted
4. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

### Environment Variables

Make sure your `.env` file contains valid API keys:
- Groq API key for LLM and audio transcription
- HuggingFace API token for embeddings

## Technical Details

- **Backend**: FastAPI with LangChain for RAG implementation
- **Frontend**: Streamlit with audio recording capabilities
- **LLM**: Groq Llama-3.3-70b-versatile
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS for document similarity search
- **Audio**: Groq Whisper-large-v3 for transcription
