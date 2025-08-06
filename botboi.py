import psutil # lets us get system info
import torch
from ingest import nomnom # calls my PDF parser
from langchain.vectorstores import FAISS #vector store library for fast similarity search
from langchain.embeddings import HuggingFaceEmbeddings # turns text into numerical vactors (embeddings) so the model can understand semantic meaning
import ollama # python wrapper for running lightwieight local chat models 
import os # standard Python OS utilities (for file paths, etc)


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


# Build vector store from training data 
def build_vector_store(file_path="training_data.txt", save_path="vector_store"):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    print(f"Embedding {len(docs)} sequences...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(save_path)
    print("Vector store saved.")
    return vectorstore


# Switch between RAG and Pure modes
def chat_with_model_hybrid(model_name, vectorstore):
    mode = "rag" # default
    
    print(f"Chate started with Ollama Model: {model_name}.")
    print(f"Commands: /rag = RAG mode, /pure = Pure mode, /exit = quit")
    print(f"Starting in {mode.upper()} mode.")

    while True:
        query = input("You: ")

        if query.lower() == "/exit":
            break
        elif query.lower() == "/rag":
            mode = "rag"
            print("Switched to RAG mode.")
            continue
        elif query.lower() == "/pure":
            mode = "pure"
            print("Switched to Pure mode.")
            continue

        if mode == "rag":
            # get relevant context from PDFs
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        else:
            # pure chat prompt
            prompt = query
        
        # Ask Ollama
        response = ollama.chat(model=model_name, messages=[{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": "prompt"}])

        print(f"AI: {respones['message']['content']}")


#Main Program Flow
def main():

    # suggest and select model
    suggested = suggest_model()
    print(f"Suggested model: {suggested}")
    choice = input(f"Press Enter to acept or type another model name: ") or suggested

    ensure_model_exists(choice)

    # Run in the chosen mode
    nomnom()
    vectorstore = build_vector_store()

    # hybrid chat sesh
    chat_with_model_hybrid(choice, vectorstore)

if __name__ == "__main__":
    main()

