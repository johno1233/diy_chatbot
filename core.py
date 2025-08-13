import os
import psutil
import torch
import ollama
from ingest import nomnom
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_PATH = "vector_store"

# Suggest a model
def suggest_model():
    # CPU and RAM check
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = psutil.cpu_count(logical=True)

    # GPU Check
    gpu_available = torch.cuda.is_available()
    gpu_vram_gb = 0
    if gpu_available:
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = gpu_props.total_memory / (1024 ** 3)
        

    print(f"Detected: {total_ram_gb:.2f} GB RAM, {cpu_count} CPU Cores.")
    if gpu_available:
        print(f"GPU: {torch.cuda.get_device_name(0)} with {gpu_vram_gb:.2f} GB VRAM")
    else:
        print("No GPU detected. Running on CPU only.")

    # Suggest a model
    if gpu_available:
        if gpu_vram_gb < 4:
            return "llama2"
        elif gpu_vram_gb < 8:
            return "mistral"
        else:
            return "phi"

# Ensure model is available locally
def ensure_model_exists(model_name):
    
    try:
        # get a list of installed models
        installed_info = ollama.list()
        installed_models = [m["model"] for m in installed_info.get("models", [])]

        # Check if requested model is installed
        if not any (model_name.lower() in m.lower() for m in installed_models):
            print(f"Model {model_name} not found locally. Pulling from Ollama...")
            ollama.pull(model_name)
            print(f"Model {model_Name} downloaded successfully.")
        else:
            print(f"Model {model_name} is already installed.")
    except Exception as e:
        print(f"Error checking Ollama models: {e}")
        print("Proceeding to pull the model anyway...")
        ollama.pull(model_name)

# Build vector store
def build_vector_store(file_path="training_data.txt", save_path="vector_store"):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    print(f"Embedding {len(docs)} sequences...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(save_path)
    print("Vector store saved.")
    return vectorstore

# Load vector store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Ollama Chat
def ollama_chat(model_name, messages):
    return ollama.chat(model=model_name, messages=messages)["message"]["content"]


suggest_model()
