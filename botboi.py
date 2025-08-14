from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import track
from rich.status import Status

from core import suggest_model, ensure_model_exists, build_vector_store, load_vector_store, ollama_chat
from ingest import nomnom, check_dir

console = Console()

# Switch between RAG and Pure modes
def chat_with_model_hybrid(model_name, vectorstore):
    mode = "rag" # default
    rag_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using provided context."}]
    pure_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using general knowledge."}] 

    console.print(Panel.fit(f"[bold cyan]CLI AI Assistant[/bold cyan]\nModel: {model_name}\nMode: {mode.upper()}", border_style="cyan"))
    console.print(Panel.fit(f"[bold yellow]Commands:\n/rag = RAG mode\n/pure = Pure mode\n/exit = quit[/bold yellow]", border_style="yellow"))
    print(f"Starting in {mode.upper()} mode.")

    while True:
        query = Prompt.ask("[bold green]You: [/bold green]")

        if query.lower() == "/exit":
            console.print("[bold red]Exiting chat...[/bold red]")
            break
        elif query.lower() == "/rag":
            mode = "rag"
            console.print("[bold cyan]Switched to RAG mode.[/bold cyan]")
            continue
        elif query.lower() == "/pure":
            mode = "pure"
            console.print("[bold cyan]Switched to Pure mode.[/bold cyan]")
            continue

        if mode == "rag":
            # get relevant context from PDFs
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {query}"
            rag_messages.append({"role": "user", "content": prompt})
            ai_reply = ollama_chat(model_name, rag_messages)
            rag_messages.append({"role": "assistant", "content": ai_reply})
        else:
            # pure chat prompt
            pure_messages.append({"role": "user", "content": query})
            ai_reply = ollama_chat(model_name, pure_messages)
            pure_messages.append({"role": "assistant", "content": ai_reply})
        
        console.print(Panel(ai_reply, title="[bold magenta]AI[bold magenta]", border_style="magenta"))


#Main Program Flow
def main():
    # Detect model
    console.print("[bold cyan]Detecting best model for your system...[/bold cyan]")
    compatible_models, selected_model = suggest_model()
    #choice = Prompt.ask("Press enter to accept model", default=selected_model)

    # Ensure model exists
    console.print("[bold cyan]Ensuring model is installed...[/bold cyan]")
    ensure_model_exists(selected_model)

    # Ask for PDF Directory
    pdf_dir = Prompt.ask("[bold cyan]Enter the full path to your PDF directory[/bold cyan]")

    # Parse PDFs with progress
    files = check_dir(pdf_dir)
    if not files:
        return
    nomnom(pdf_dir) 

    # Bulding vector store with spinner
    with Status("[bold green] Building vector store...[/bold green]", spinner="dots"):
        vectorstore = build_vector_store()

    console.print("[bold green]Model and data preparation complete![/bold green]")

    # Start chat
    chat_with_model_hybrid(selected_model, vectorstore)

if __name__ == "__main__":
    main()

