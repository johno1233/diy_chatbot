from core import suggest_model, ensure_model_exists, build_vector_store, load_vector_store, ollama_chat
from ingest import nomnom

# Switch between RAG and Pure modes
def chat_with_model_hybrid(model_name, vectorstore):
    mode = "rag" # default
    rag_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using provided context."}]
    pure_messages = [{"role": "system", "content": "You are a helpful AI assistant that answers questions using general knowledge."}] 

    print(f"Chat started with Ollama Model: {model_name}.")
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

