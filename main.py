import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from llama_index.core import (
    VectorStoreIndex, 
    Settings, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.memory import ChatMemoryBuffer # <--- CRITICAL

load_dotenv()

# --- 1. CONFIGURATION ---
STORAGE_DIR = "./storage"
DATA_DIR = "./data"

memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

system_prompt = """
    You are a helpful educational assistant.
    Base answers ONLY on existing documents received and chat history.
    Think before answering.
    You can FAIL if you do not know the answer.
"""

Settings.llm = Groq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    device="cpu"
)

# --- 2. PERSISTENCE LOGIC ---
def get_index():
    if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    else:
        return rebuild_index()

def rebuild_index():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        return VectorStoreIndex.from_documents([])

    docling_reader = DoclingReader()
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        file_extractor={
            ".pdf": docling_reader,
            ".docx": docling_reader,
            ".pptx": docling_reader,
            ".html": docling_reader
        }
    )
    
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

# --- 3. API IMPLEMENTATION ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(
    message: str = Form(...),
    file: UploadFile = File(None)
):
    # Step 1: Update documents if a new file is sent
    if file:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        index = rebuild_index()
    else:
        index = get_index()

    # Step 2: Initialize Chat Engine with Global Memory
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        system_prompt=system_prompt
    )

    # Step 3: Generate Response
    response = chat_engine.chat(message)
    
    return {"answer": str(response)}

@app.post("/reset")
async def reset_memory():
    """Endpoint to manually clear the conversation history."""
    memory.reset()
    return {"status": "Chat history cleared."}

@app.post("/test")
async def test(
    message: str = Form(...),
    file: UploadFile = File(None)
):
    return {"message": message, "file": file}


if __name__ == "__main__":
    import uvicorn
    for path in [DATA_DIR, STORAGE_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    uvicorn.run(app, host="0.0.0.0", port=8000)