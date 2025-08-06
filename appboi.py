import streamlit as st
from core import suggest_model, ensure_model_exists, load_vector_store, build_vector_store, ollama_chat
from ingest import nomnom

st.set_page_config(page_title="AppBoi", layout="wide")
st.title("AI Assistant - Hybrid (RAG & Pure Chat)")


# Model Selection
default_model = suggest_model()
model_name = st.text_input("Ollama Model", default_model)

# Prepare Model and RAG Data
if st.button("Prepare Model & Data"):
    ensure_model_exists(model_name)
    nomnom()
    build_vector_store()
    st.success("Model and Data ready!")

# Session State for histories
if "rag_history" not in st.session_state:
    st.session_state.rag_history = [{"role": "system", "content": "You are a helpful AI assistant using PDF context to answer questions."}]
if "pure_history" not in st.session_state:
    st.session_state.pure_history = [{"role": "system", "sontent": "Your are a helpful AI assistant using general knowledge to answer questions"}]
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "RAG"

# Mode Switcher
st.session_state.active_mode = st.radio("Select Mode", ["RAG", "Pure Chat"])

# Chat Input
query = st.text_input("Your Message: ")

if st.button("Send") and query:
    if st.session_state.active_mode == "RAG":
        vectorstore = load_vector_store()
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        user_input = f"Answer the following question based on this context:\n\n{context}\n\nQuestion: {query}"
        st.session_state.rag_history.append({"role": "user", "content": user_input})
        reply = ollama_chat(model_name, st.session_state.rag_history)
        st.session_state.rag_history.append({"role": "assistant", "content": reply})
    else:
        st.session_state.pure_history.append({"role": "user", "content": query})
        reply = ollama_chat(model_name, st.session_state.pure_history)

# Display chat history
if st.session_state.active_mode == "RAG":
    st.subheader("RAG Chat History")
    for msg in st.session_state.rag_history:
        if msg["role"] == "user":
            st.markdown(f"You: {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"AI: {msg['content']}")
else:
    st.subheader("Pure Chat History")
    for msg in st.session_state.pure_history:
        if msg["role"] == "user":
            st.markdown(f"You: {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"AI: {msg['content']}")

