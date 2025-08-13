from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import track

from core import suggest_model, ensure_model_exists, build_vector_store, load_vector_store, ollama_chat
from ingest import nomnom

console = Console()

# Switch between RAG and Pure modes
def chat_with_model_hybrid(model_name, vectorstore):
    mode = "rag" # default
    rag_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using provided context."}]
    pure_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using general knowledge."}] 

    console.print(Panel.fit(f"[bold cyan]CLI AI Assistant[/bold cyan]\nModel: {model_name}\nMode: {mode.upper()}", border_style="cyan"))
    print(Panel.fit(f"[bold yellow]Commands: /rag = RAG mode\n/pure = Pure mode\n/exit = quit[/bold yellow]", border_style="yellow"))
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

            prompt = f"Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {query}"
            rag_messages.append({"role": "user", "content": prompt})
            ai_reply = ollama_chat(model_name, rag_messages)
            rag_messages.append({"role": "assistant", "content": ai_reply})
        else:
            # pure chat prompt
            pure_messages.append({"role": "user", "content": query})
            ai_reply = ollama_chat(model_name, pure_messages)
            pure_messages.append({"role": "assistant", "content": ai_reply})
        
        print(f"AI: {ai_reply}")


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

