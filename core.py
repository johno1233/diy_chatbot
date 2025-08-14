import os
import psutil
import torch
import ollama
from ingest import nomnom
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

VECTOR_STORE_PATH = "vector_store"

console = Console()

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
        
    # Display system specs in a styled panel
    system_info = f"[bold]RAM:[/bold] {total_ram_gb:.3f} GB\n[bold]CPU Cores:[/bold] {cpu_count}"
    if gpu_available:
        system_info += f"\n[bold]GPU:[/bold] {torch.cuda.get_device_name(0)} with {gpu_vram_gb:.2f} GB VRAM"
    else:
        system_info += "\n[bold]GPU:[/bold] None (Running on CPU only)"
    console.print(Panel(system_info, title="System Specifications", border_style="cyan"))

    # Define Ollama models with approximate Ram requiremends
    ollama_models = [
        {"name": "smollm2:135m", "params": 0.135, "min_ram": 1, "min_vram": 0, "use_case": "Lightweight tasks"},
        {"name": "smollm2:360m", "params": 0.36, "min_ram": 2, "min_vram": 0, "use_case": "Lightweight tasks"},
        {"name": "smollm2:1.7b", "params": 1.7, "min_ram": 4, "min_vram": 0, "use_case": "Lightweight general tasks"},
        {"name": "tinyllama:1.1b", "params": 1.1, "min_ram": 3, "min_vram": 0, "use_case": "Lightweight general tasks"},
        {"name": "qwen2:0.5b", "params": 0.5, "min_ram": 2, "min_vram": 0, "use_case": "Lightweight multilingual tasks"},
        {"name": "qwen2:1.5b", "params": 1.5, "min_ram": 4, "min_vram": 0, "use_case": "Lightweight multilingual tasks"},
        {"name": "gemma:2b", "params": 2, "min_ram": 5, "min_vram": 0, "use_case": "General purpose"},
        {"name": "phi3:3.8b", "params": 3.8, "min_ram": 8, "min_vram": 0, "use_case": "General purpose, efficient"},
        {"name": "llama2:7b", "params": 7, "min_ram": 12, "min_vram": 4, "use_case": "General purpose"},
        {"name": "mistral:7b", "params": 7, "min_ram": 12, "min_vram": 4, "use_case": "General purpose, efficient"},
        {"name": "codellama:7b", "params": 7, "min_ram": 12, "min_vram": 4, "use_case": "Code generation"},
        {"name": "llama3:8b", "params": 8, "min_ram": 16, "min_vram": 6, "use_case": "Advanced reasoning"},
        {"name": "qwen2:7b", "params": 7, "min_ram": 12, "min_vram": 4, "use_case": "Multilingual tasks"},
        {"name": "gemma2:9b", "params": 9, "min_ram": 16, "min_vram": 6, "use_case": "General purpose"},
        {"name": "llama2:13b", "params": 13, "min_ram": 24, "min_vram": 8, "use_case": "General purpose"},
        {"name": "codellama:13b", "params": 13, "min_ram": 24, "min_vram": 8, "use_case": "Code generation"},
        {"name": "mistral-nemo:12b", "params": 12, "min_ram": 24, "min_vram": 8, "use_case": "Advanced reasoning"},
        {"name": "phi4:14b", "params": 14, "min_ram": 28, "min_vram": 10, "use_case": "General purpose, efficient"},
        {"name": "llama3.1:8b", "params": 8, "min_ram": 16, "min_vram": 6, "use_case": "Advanced reasoning"},
        {"name": "mixtral:8x7b", "params": 46, "min_ram": 64, "min_vram": 24, "use_case": "High-performance tasks"},
        {"name": "llama3:70b", "params": 70, "min_ram": 128, "min_vram": 40, "use_case": "Enterprise-grade tasks"},
    ]

    # Filter compatible models based on system specs
    compatible_models = []
    for model in ollama_models:
        ram_ok = total_ram_gb >= model["min_ram"]
        vram_ok = gpu_vram_gb >= model["min_vram"] if model["min_vram"] > 0 else True
        if ram_ok and vram_ok:
            compatible_models.append(model)

    # Sort models by parameter size (ascending) for suggestion logic
    compatible_models.sort(key=lambda x: x["params"])

    # Suggest a model: prefer larger models for better performance if resources allow
    suggested_model = None
    if compatible_models:
        # If GPU available and VRAM >= 10GB, suggest a high-performance model
        if gpu_available and gpu_vram_gb >= 10:
            for model in reversed(compatible_models): # Prefer larger models
                if model["min_vram"] <= gpu_vram_gb:
                    suggested_model = model["name"]
                    break
        # If GPU available but Vram < 10 GB, suggest an efficient model
        elif gpu_available:
            for model in compatible_models:
                if model["min_vram"] <= gpu_vram_gb and model["name"] in ["mistral:7b", "phi3:3.8b", "llama3.1:8b"]:
                    suggested_model = model["name"]
                    break
        # CPU-only, suggest a lightweight model
        else:
            for model in compatible_models:
                if model["min_vram"] == 0 and model["name"] in ["phi3:3.8b", "smollm2:1.7b", "qwen2:1.5b"]:
                    suggested_model = model["name"]
                    break
        # Fallback to smallest compatible model if no preferred model found
        if not suggested_model and compatible_models:
            suggested_model = compatible_models[0]["name"]

    if not compatible_models:
        console.print("[bold red]No compatible models found for you system specifications.[/bold red]")
        return [], None

    # Display compatible models in a table
    table = Table(title="Compatible Models", title_style="bold magenta", border_style="cyan")
    table.add_column("No.", style="cyan", justify="center")
    table.add_column("Model Name", style="green")
    table.add_column("Parameters", style="yellow")
    table.add_column("Use Case", style="blue")
    table.add_column("Min RAM (GB)", style="white")
    table.add_column("Min VRAM (GB)", style="white")

    for i, model in enumerate(compatible_models, 1):
        table.add_row(
            str(i),
            model["name"],
            f"{model['params']}B",
            model["use_case"],
            str(model["min_ram"]),
            str(model["min_vram"]) if model["min_vram"] > 0 else "N/A"
        )
    
    console.print(table)
    console.print(Panel(f"[bold green]Suggested Model: {suggested_model}[/bold green]", border_style="green"))

    # Allow user to choose a model
    while True:
        choice = Prompt.ask("\nEnter the number of the model to use (or press Enter to use the suggested model)", 
            default="")
        if choice == "":
            console.print(f"[bold green]Using suggested model: {suggested_model}[/bold green]")
            return [model["name"] for model in compatible_models], suggested_model
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(compatible_models):
                selected_model = compatible_models[choice_idx]["name"]
                console.print(f"[bold green]Selected Model: {selected_model}[/bold green]")
                return [model["name"] for model in compatible_models], selected_model
            else:
                console.print(f"[bold red]Please enter a number between 1 and {len(compatible_models)}.[/bold red]")
        except ValueError:
            console.print("[bold red]Invalid input. Please enter a valid number or press Enter for the suggested model.[/bold red]")


# Ensure model is available locally
def ensure_model_exists(model_name):
    
    try:
        # get a list of installed models
        installed_info = ollama.list()
        installed_models = [m["model"] for m in installed_info.get("models", [])]

        # Check if requested model is installed
        if not any (model_name.lower() in m.lower() for m in installed_models):
            console.print(f"[bold yellow]Model {model_name} not found locally. Pulling from Ollama...[/bold yellow]")
            ollama.pull(model_name)
            console.print(f"[bold green]Model {model_Name} downloaded successfully.[/bold green]")
        else:
            print(f"Model {model_name} is already installed.")
    except Exception as e:
        console.print(f"[bold red]Error checking Ollama models: {e}[/bold red]")
        console.print("[bold yellow]Proceeding to pull the model anyway...[/bold yellow]")
        ollama.pull(model_name)

# Build vector store
def build_vector_store(file_path="training_data.txt", save_path="vector_store"):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    console.print(f"[bold cyan]Embedding {len(docs)} sequences...[/bold cyan]")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(save_path)
    console.print("[bold green]Vector store saved.[/bold green]")
    return vectorstore

# Load vector store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Ollama Chat
def ollama_chat(model_name, messages):
    return ollama.chat(model=model_name, messages=messages)["message"]["content"]


