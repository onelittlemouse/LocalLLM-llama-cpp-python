import os
import time
import threading
import logging
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional
from fastapi.testclient import TestClient
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Model configuration
MODEL_PATH = "E:/LLM/models/qwen/qwen2_5/qwen2.5-3b-instruct-q8_0.gguf"  # Update with your GGUF model path
CONTEXT_LENGTH = 16384
CPU_THREADS = 24
GPU_LAYERS = -1
VERBOSE = False
MODEL_PARAMS = {
    "n_ctx": CONTEXT_LENGTH,  # Context length
    "n_threads": CPU_THREADS,  # Number of CPU threads to use
    "n_gpu_layers": GPU_LAYERS,  # Number of layers to offload to GPU (if available)
    "verbose": VERBOSE  # Whether to print debug info
}
MAX_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 40
MIN_P = 0.05
STOP = None
ECHO = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Disable print INFO

# Initialize local gguf llm
def initialize_model():
    """Initialize the LLM model"""
    global llm
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logger.info(f"Loading model from {MODEL_PATH}...")
    llm = Llama(model_path=MODEL_PATH, **MODEL_PARAMS)
    logger.info("Model loaded successfully")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model when the application starts"""
    try:
        initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
    yield
    pass

app = FastAPI(
    title="Local LLM API",
    description="API for interacting with locally hosted gguf LLM using llama.cpp",
    version="0.0.1",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    min_p: float = MIN_P
    stop: Optional[List[str]] = STOP
    echo: bool = ECHO


class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    min_p: float = MIN_P
    stop: Optional[List[str]] = STOP


class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


@app.get("/v1/models")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": llm is not None}


@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    """Generate completion for a given prompt"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        response = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            stop=request.stop,
            echo=request.echo
        )
        return {
            "completion": response["choices"][0]["text"],
            "usage": {
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"]
            }
        }
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """Generate chat completion for a conversation"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Format messages for the model
        formatted_messages = "\n".join(
            f"{msg.role}: {msg.content}" for msg in request.messages
        )
        prompt = f"{formatted_messages}\nassistant:"
        response = llm(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            stop=request.stop
        )
        return {
            "message": {
                "role": "assistant",
                "content": response["choices"][0]["text"]
            },
            "usage": {
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"]
            }
        }
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embedding(request: EmbeddingRequest):
    """Generate embeddings for the input texts"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        embeddings = []
        for text in request.texts:
            # Get embedding vector
            embedding = llm.create_embedding(text)['data'][0]['embedding']
            # Convert to numpy array for potential normalization
            embedding_array = np.array(embedding)
            if request.normalize:
                # Normalize to unit vector (L2 norm)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm
            embeddings.append(embedding_array.tolist())
        return {
            "embeddings": embeddings,
            "embedding_size": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def test_completion():
    """Test function to demonstrate text completion"""
    client = TestClient(app)
    print("Start text completion!")
    prompt = input("Text completion:")
    response = client.post(
        "/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_P": MIN_P,
            "stop": STOP,
            "echo": ECHO
        }
    )
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=4))


def test_chat_completion():
    """Test function to demonstrate chat completion"""
    client = TestClient(app)
    # Example conversation
    print("\nStart chat completion!")
    system_prompt = input("System:")
    user_requirement = input("User:")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_requirement}
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_P": MIN_P,
            "stop": STOP
        }
    )
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=4))


def test_infn_turns_chat():
    """Test function for infinite chat conversation in the console"""
    client = TestClient(app)
    # Initialize conversation with system message
    messages = [
        {"role": "system", "content": "You are a helpful, knowledgeable AI assistant."}
    ]
    print("\nStarting infinite chat (type 'quit' to exit)!")
    print("System prompt:", messages[0]["content"])
    print("=" * 50)
    try:
        while True:
            # Get user input
            user_input = input("User: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Ending chat...")
                break
            if not user_input.strip():
                print("Please enter a message!")
                continue
            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})
            # Get AI response
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "min_P": MIN_P,
                    "stop": STOP
                }
            )
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                continue
            llm_response = response.json()["message"]["content"]
            # Add AI response to conversation history
            messages.append({"role": "assistant", "content": llm_response})
            # Print AI response with some formatting
            print(f"\nAssistant:\n{llm_response}")
    except KeyboardInterrupt:
        print("\nChat ended by user!")
    except Exception as e:
        print(f"Error during chat: {str(e)}")


def test_embedding():
    """Test function to demonstrate embedding generation"""
    client = TestClient(app)
    texts = [
        "The capital of France is Paris",
        "Paris is located in northern France",
        "Quantum computing uses qubits instead of bits"
    ]
    response = client.post(
        "/embeddings",
        json={
            "texts": texts,
            "normalize": True
        }
    )
    print("\nEmbeddings Test:")
    print(f"Status Code: {response.status_code}")
    print("Response:")
    result = response.json()
    # Print the shape and first few dimensions of each embedding
    for i, (text, embedding) in enumerate(zip(texts, result["embeddings"])):
        print(f"\nText {i + 1}: {text[:50]}...")
        print(f"Embedding size: {len(embedding)}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Norm: {np.linalg.norm(np.array(embedding)):.4f}")



@app.get("/v1/chat-ui", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat interface HTML page"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Local LLM Chat</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .chat-container {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .chat-box {{
                height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .message {{
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 5px;
            }}
            .user-message {{
                background-color: #e3f2fd;
                margin-left: 20%;
            }}
            .assistant-message {{
                background-color: #f1f1f1;
                margin-right: 20%;
                white-space: pre-wrap;
            }}
            .system-message {{
                background-color: #e8f5e9;
                font-style: italic;
                font-size: 0.9em;
            }}
            #message-input {{
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 10px;
            }}
            .button-group {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }}
            button {{
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .lang-btn {{
                background-color: #2196F3;
            }}
            .clear-btn {{
                background-color: #f44336;
            }}
        </style>
    </head>
    <body>
        <h1>Local LLM Chat</h1>
        <div class="chat-container">
            <div class="button-group">
                <button onclick="setLanguage('en')" class="lang-btn">English</button>
                <button onclick="setLanguage('zh-cn')" class="lang-btn">中文</button>
                <button onclick="clearChat()" class="clear-btn">Clear Chat</button>
            </div>

            <div class="chat-box" id="chat-box"></div>

            <textarea id="message-input" placeholder="Type your message here..." rows="3"></textarea>
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            let currentLanguage = 'en';
            let messages = [];
            // System prompts for different languages
            const systemPrompts = {{
                'en': 'You are a helpful, knowledgeable AI assistant. Respond in English.',
                'zh-cn': '你是一个乐于助人、知识渊博的AI助手。请用中文回答。'
            }};
            function setLanguage(lang) {{
                currentLanguage = lang;
                const input = document.getElementById('message-input');
                if (lang === 'en') {{
                    input.placeholder = 'Type your message here...';
                }} else {{
                    input.placeholder = '在此输入您的消息...';
                }}
                addSystemMessage(systemPrompts[lang]);
            }}
            function addSystemMessage(content) {{
                const chatBox = document.getElementById('chat-box');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message system-message';
                messageDiv.innerHTML = '<strong>System:</strong> ' + escapeHtml(content);
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
                // Add to messages array
                messages.push({{
                    role: 'system',
                    content: content
                }});
            }}
            function addMessage(role, content) {{
                const chatBox = document.getElementById('chat-box');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{role}}-message`;
                if (role === 'user') {{
                    messageDiv.innerHTML = '<strong>User:</strong> ' + escapeHtml(content);
                }} else {{
                    // Add newlines for AI responses
                    const formattedContent = escapeHtml(content).replace(/\\n/g, '<br>');
                    messageDiv.innerHTML = '<strong>Assistant:</strong><br>' + formattedContent;
                }}
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }}
            // Helper function to escape HTML
            function escapeHtml(unsafe) {{
                return unsafe
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;")
                    .replace(/"/g, "&quot;")
                    .replace(/'/g, "&#039;");
            }}
            async function sendMessage() {{
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                // Add user message to UI
                addMessage('user', message);
                // Add to messages array
                messages.push({{
                    role: 'user',
                    content: message
                }});
                // Clear input
                input.value = '';
                try {{
                    // Show loading indicator
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'message assistant-message';
                    loadingDiv.innerHTML = '<em>Assistant is thinking...</em>';
                    document.getElementById('chat-box').appendChild(loadingDiv);
                    // Send to API
                    const response = await fetch('/v1/chat/completions', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            messages: messages,
                            max_tokens: {MAX_TOKENS},
                            temperature: {TEMPERATURE},
                            top_p: {TOP_P},
                            top_k: {TOP_K},
                            min_p: {MIN_P},
                            stop: {json.dumps(STOP) if STOP else 'null'}
                        }})
                    }});
                    // Remove loading indicator
                    document.getElementById('chat-box').removeChild(loadingDiv);
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    const data = await response.json();
                    const assistantMessage = data.message.content;
                    // Add assistant message to UI
                    addMessage('assistant', assistantMessage);
                    // Add to messages array
                    messages.push({{
                        role: 'assistant',
                        content: assistantMessage
                    }});
                }} catch (error) {{
                    console.error('Error:', error);
                    addMessage('assistant', `Error: ${{error.message}}`);
                }}
            }}
            function clearChat() {{
                document.getElementById('chat-box').innerHTML = '';
                messages = [];
                addSystemMessage(systemPrompts[currentLanguage]);
            }}
            // Initialize with English
            setLanguage('en');
            // Handle Enter key
            document.getElementById('message-input').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage();
                }}
            }});
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    # Start the server in a separate thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "0.0.0.0", "port": 1234},
        daemon=True
    )
    server_thread.start()
    print("Waiting for the server thread...")
    time.sleep(15)

    # Run the test functions
    try:
        # test_completion()
        # test_chat_completion()
        test_infn_turns_chat()
        # test_embedding()
    except Exception as e:
        print(f"Error during testing: {e}")

    # Keep the server running (press Ctrl+C to stop)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_thread.join()