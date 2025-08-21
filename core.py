import os
import psutil
import torch
import ollama
import requests
from ingest import nomnom
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

VECTOR_STORE_PATH = "vector_store"

console = Console()


def fetch_remote_models():
    try:
        resp = requests.get("https://ollama.com/library", timeout=10)
        if resp.status_code == 200:
            soup = BeaturifulSoup(resp.text, 'html.parser')
            models = []
            for a in soup.find_all('a', href=lambda h: h an h.startswith('/library') and len(h.split('/')) == 3):
                slug = a['href'].split('/')[-1]
                h2 = a.find('h2')
                if h2:
                    name = h2.text.strip()
                    p = a.find('p')
                    description = p.text.strip() if p else "No description available"
                    models.append({
                        "name": slug,
                        "param_param_size": "?",
                        "min_ram": "?",
                        "min_vram": "?",
                        "use_case": description,
                        "source": "Remote",
                    })
            if models:
                return models
            else:
                console.print("[bold yellow]No remote models found on the library page.[/bold yellow]")
        else:
            console.print(f"[bold red]Failed ot fetch remote moedls (status {resp.status_code})[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error fetching remote models: {e}[/bold red]")
    return []

# Suggest a model
def suggest_model():
    # CPU and RAM check
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count(logical=True)

    # GPU Check
    gpu_available = torch.cuda.is_available()
    gpu_vram_gb = 0
    if gpu_available:
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = gpu_props.total_memory / (1024**3)

    # Display system specs in a styled panel
    system_info = (
        f"[bold]RAM:[/bold] {total_ram_gb:.1f} GB\n[bold]CPU Cores:[/bold] {cpu_count}"
    )
    if gpu_available:
        system_info += f"\n[bold]GPU:[/bold] {torch.cuda.get_device_name(0)} with {gpu_vram_gb:.1f} GB VRAM"
    else:
        system_info += "\n[bold]GPU:[/bold] None (Running on CPU only)"
    console.print(
        Panel(system_info, title="System Specifications", border_style="cyan")
    )

    # Define Ollama models with approximate Ram requirements
    # Query ollama for installed models
    try:
        installed_info = ollama.list()
        ollama_models = installed_info.get("models", [])
    except Exception as e:
        console.print(f"[bold red]Error fetching models from Ollama: {e}[/bold red]")
        return [], None

    if not ollama_models:
        console.print(
            "[bold red]No Ollama models found. Please pull one with 'ollama pull <model>'[/bold red]"
        )
        return [], None

    # Filter compatible models based on system specs
    compatible_models = []
    for m in ollama_models:
        name = m.get("name") or m.get("model") or "unknown"
        size_bytes = m.get("size", 0)
        size_gb = size_bytes / (1024**3) if size_bytes else 0

        param_size = m.get("details", {}).get("parameter_size", f"{size_gb:.1f} GB est")

        min_ram = max(2, size_gb * 2)
        min_vram = 0 if not gpu_available else min(size_gb * 0.5, gpu_vram_gb)

        if total_ram_gb >= min_ram:
            compatible_models.append(
                {
                    "name": name,
                    "param_size": param_size,
                    "min_ram": round(min_ram, 1),
                    "min_vram": round(min_vram, 1),
                    "use_case": "General Purpose",  # will expand later
                    "source": "Installed",
                }
            )

    remote_models = fetch_remote_models()

    all_models = compatible_models + remote_models

    if not all_models:
        console.print("[bold red]No models found (local or remote).[/bold red]")
        return [], None
    
    # Sort models by parameter size (ascending) for suggestion logic
    def sort_key(x):
        ps = x["param_size"]
        if ps == "?":
            return 0
        try:
            return float(ps.strip(' GB est').rstrip('B'))
        except ValueError:
            return 0

    all_models.sort(key=sort_key)

    # Display compatible models in a table
    table = Table(
        title="Compatible Models (Local + Remote)",
        title_style="bold magenta",
        border_style="cyan",
    )
    table.add_column("No.", style="cyan", justify="center")
    table.add_column("Model Name", style="green")
    table.add_column("Parameters", style="yellow")
    table.add_column("Use Case", style="blue")
    table.add_column("Min RAM (GB)", style="white")
    table.add_column("Min VRAM (GB)", style="white")
    table.add_column("Source", style="white")

    for i, model in enumerate(all_models, 1):
        table.add_row(
            str(i),
            model["name"],
            model["param_size"],
            model["use_case"],
            str(model["min_ram"]),
            str(model["min_vram"]) if model["min_vram"] > 0 else "N/A",
            model["source"],
        )

    console.print(table)
    suggested_model = (
        compatible_models[-1]["name"] if compatible_models else all_models[0]["name"]
    )
    console.print(
        Panel(
            f"[bold green]Suggested Model: {suggested_model}[/bold green]",
            border_style="green",
        )
    )

    # Allow user to choose a model
    while True:
        choice = Prompt.ask(
            "\nEnter the number of the model to use (or press Enter to use the suggested model)",
            default="",
        )
        if choice == "":
            console.print(
                f"[bold green]Using suggested model: {suggested_model}[/bold green]"
            )
            return [model["name"] for model in compatible_models], suggested_model
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(all_models):
                selected_model = all_models[choice_idx]["name"]
                source = all_models[choice_idx]["source"]
                if source == "Remote":
                    console.print(
                        f"[bold yellow]Model {selected_model} is not installed. Pulling from Ollama Hub...[/bold yellow]"
                    )
                    os.system(f"ollama pull {selected_model}")
                console.print(
                    f"[bold green]Selected Model: {selected_model}[/bold green]"
                )
                return [model["name"] for model in all_models], selected_model
            else:
                console.print(
                    f"[bold red]Please enter a number between 1 and {len(compatible_models)}.[/bold red]"
                )
        except ValueError:
            console.print(
                "[bold red]Invalid input. Please enter a valid number or press Enter for the suggested model.[/bold red]"
            )


# Ensure model is available locally
def ensure_model_exists(model_name):
    try:
        # get a list of installed models
        installed_info = ollama.list()
        installed_models = [m["model"] for m in installed_info.get("models", [])]

        # Check if requested model is installed
        if not any(model_name.lower() in m.lower() for m in installed_models):
            console.print(
                f"[bold yellow]Model {model_name} not found locally. Pulling from Ollama...[/bold yellow]"
            )
            ollama.pull(model_name)
            console.print(
                f"[bold green]Model {model_name} downloaded successfully.[/bold green]"
            )
        else:
            print(f"Model {model_name} is already installed.")
    except Exception as e:
        console.print(f"[bold red]Error checking Ollama models: {e}[/bold red]")
        console.print(
            "[bold yellow]Proceeding to pull the model anyway...[/bold yellow]"
        )
        ollama.pull(model_name)


# Build vector store
def build_vector_store(file_path="training_data.txt", save_path="vector_store"):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    console.print(f"[bold cyan]Embedding {len(docs)} sequences...[/bold cyan]")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(save_path)
    console.print("[bold green]Vector store saved.[/bold green]")
    return vectorstore


# Load vector store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )


# Ollama Chat
def ollama_chat(model_name, messages):
    return ollama.chat(model=model_name, messages=messages)["message"]["content"]
