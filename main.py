import os
import tempfile
import hashlib
import io
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# ---- Document Chatbot Core ----
class DocumentChatbot:
    def __init__(self):
        # Initialize embeddings
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )
        
        # Initialize Groq client for audio transcription
        self.groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Prompt templates
        self.document_template = """You are an intelligent document assistant. Answer questions based on the provided context from the uploaded document.
If the context doesn't contain relevant information to answer the question, politely say you cannot find that information in the document.
Be helpful, accurate, and cite specific parts of the document when possible.

User Question: {user_input}

Document Context:
{context}

Chat History:
{history}

Provide a clear and helpful response based on the document content."""
        
        self.general_template = """You are a helpful AI assistant. Answer the user's question to the best of your ability.
Be conversational, helpful, and informative.

User Question: {user_input}

Chat History:
{history}

Provide a helpful response."""
        
        # Storage for document databases (session-based)
        self.document_stores = {}
        self.general_chat_history = []
    
    def process_pdf(self, file_content: bytes, filename: str) -> str:
        """Process PDF content and create vector store"""
        try:
            # Create a hash for the file to use as session ID
            file_hash = hashlib.md5(file_content).hexdigest()
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                if not documents:
                    raise ValueError("No content found in PDF")
                
                # Split documents
                texts = self.text_splitter.split_documents(documents)
                
                if not texts:
                    raise ValueError("No text content extracted from PDF")
                
                # Create vector store
                vectorstore = FAISS.from_documents(texts, self.embeddings)
                
                # Store in memory with file hash as key
                self.document_stores[file_hash] = {
                    'vectorstore': vectorstore,
                    'filename': filename,
                    'doc_count': len(texts)
                }
                
                return file_hash
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def transcribe_audio(self, audio_data) -> str:
        """Transcribe audio file using Groq Whisper"""
        try:
            # Handle different types of audio input
            if hasattr(audio_data, 'read'):
                # It's a file-like object
                audio_bytes = audio_data.read()
                audio_data.seek(0)  # Reset file pointer
            else:
                # It's already bytes
                audio_bytes = audio_data
            
            # Create a proper file-like object for Groq
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # Groq needs a filename
            
            transcription = self.groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text"
            )
            return transcription
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    def query_document(self, session_id: str, user_input: str, chat_history: list) -> str:
        """Query the document with user input"""
        if session_id not in self.document_stores:
            return "No document found for this session. Please upload a document first."
        
        try:
            # Get vector store
            vectorstore = self.document_stores[session_id]['vectorstore']
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # Retrieve relevant context
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Format chat history (last 3 exchanges)
            last_3 = chat_history[-3:] if len(chat_history) > 3 else chat_history
            history_str = "\n".join([f"User: {u}\nAssistant: {b}" for u, b in last_3])
            
            # Create prompt
            prompt = self.document_template.format(
                user_input=user_input,
                context=context,
                history=history_str
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            return f"Error querying document: {str(e)}"
    
    def general_chat(self, user_input: str) -> str:
        """General chat without document context"""
        try:
            # Format chat history (last 5 exchanges)
            last_5 = self.general_chat_history[-5:] if len(self.general_chat_history) > 5 else self.general_chat_history
            history_str = "\n".join([f"User: {u}\nAssistant: {b}" for u, b in last_5])
            
            # Create prompt
            prompt = self.general_template.format(
                user_input=user_input,
                history=history_str
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Update general chat history
            self.general_chat_history.append((user_input, response.content))
            
            return response.content
            
        except Exception as e:
            return f"Error in general chat: {str(e)}"
    
    def get_document_info(self, session_id: str) -> Optional[dict]:
        """Get information about the uploaded document"""
        if session_id in self.document_stores:
            return {
                'filename': self.document_stores[session_id]['filename'],
                'doc_count': self.document_stores[session_id]['doc_count']
            }
        return None
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.document_stores:
            del self.document_stores[session_id]

# ---- FastAPI App ----
app = FastAPI(title="Document Assistant Chatbot API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = DocumentChatbot()

# In-memory storage for chat histories
chat_histories = {}

# ---- Request & Response Models ----
class ChatRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None  # None for general chat

class ChatResponse(BaseModel):
    bot_response: str
    history: List[Tuple[str, str]]
    document_info: Optional[dict] = None

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    message: str
    doc_count: int

class TranscriptionResponse(BaseModel):
    transcription: str

# ---- Endpoints ----

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the PDF
        session_id = chatbot.process_pdf(content, file.filename)
        
        # Initialize chat history for this session
        chat_histories[session_id] = []
        
        # Get document info
        doc_info = chatbot.get_document_info(session_id)
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            message="Document uploaded and processed successfully",
            doc_count=doc_info['doc_count']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the uploaded document or general chat"""
    try:
        user_input = request.user_input
        session_id = request.session_id
        
        if session_id:
            # Document-based chat
            if session_id not in chat_histories:
                raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
            
            # Get chat history for this session
            history = chat_histories[session_id]
            
            # Query the document
            bot_response = chatbot.query_document(session_id, user_input, history)
            
            # Update history
            chat_histories[session_id].append((user_input, bot_response))
            
            # Get document info
            document_info = chatbot.get_document_info(session_id)
            
            return ChatResponse(
                bot_response=bot_response,
                history=chat_histories[session_id],
                document_info=document_info
            )
        else:
            # General chat
            bot_response = chatbot.general_chat(user_input)
            
            return ChatResponse(
                bot_response=bot_response,
                history=chatbot.general_chat_history,
                document_info=None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file to text"""
    try:
        # Validate audio file type
        allowed_types = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/webm']
        if audio.content_type not in allowed_types and not any(audio.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.webm']):
            raise HTTPException(status_code=400, detail="Unsupported audio format. Please use MP3, WAV, M4A, or WebM.")
        
        # Read audio content
        audio_content = await audio.read()
        
        # Create a proper BytesIO object with filename
        audio_file = io.BytesIO(audio_content)
        audio_file.name = audio.filename or "audio.wav"
        
        # Transcribe audio
        transcription = chatbot.transcribe_audio(audio_file)
        
        return TranscriptionResponse(transcription=transcription)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.post("/chat-audio")
async def chat_with_audio(
    session_id: Optional[str] = Form(None),
    audio: UploadFile = File(...)
):
    """Chat using audio input"""
    try:
        # Read audio content
        audio_content = await audio.read()
        
        # Create a proper BytesIO object
        audio_file = io.BytesIO(audio_content)
        audio_file.name = audio.filename or "audio.wav"
        
        # First transcribe the audio
        transcription = chatbot.transcribe_audio(audio_file)
        
        # Then process the chat
        chat_request = ChatRequest(user_input=transcription, session_id=session_id)
        chat_response = await chat_endpoint(chat_request)
        
        return {
            "transcription": transcription,
            "bot_response": chat_response.bot_response,
            "history": chat_response.history,
            "document_info": chat_response.document_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio chat: {str(e)}")

@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get information about a session"""
    document_info = chatbot.get_document_info(session_id)
    if not document_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history_count = len(chat_histories.get(session_id, []))
    
    return {
        "session_id": session_id,
        "document_info": document_info,
        "chat_history_count": history_count
    }

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    try:
        # Clear from chatbot
        chatbot.clear_session(session_id)
        
        # Clear chat history
        if session_id in chat_histories:
            del chat_histories[session_id]
        
        return {"message": "Session cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.post("/clear-general-chat")
async def clear_general_chat():
    """Clear general chat history"""
    try:
        chatbot.general_chat_history = []
        return {"message": "General chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing general chat: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Document Assistant API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Assistant Chatbot API",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload a PDF document",
            "chat": "POST /chat - Chat with uploaded document or general chat",
            "transcribe": "POST /transcribe - Transcribe audio to text",
            "chat-audio": "POST /chat-audio - Chat using audio input",
            "session-info": "GET /sessions/{session_id}/info - Get session information",
            "clear-session": "DELETE /sessions/{session_id} - Clear a session",
            "clear-general-chat": "POST /clear-general-chat - Clear general chat history",
            "health": "GET /health - Health check"
        }
    }

# Run with: uvicorn backend:app --reload --host 127.0.0.1 --port 8000
