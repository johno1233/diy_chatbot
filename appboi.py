import streamlit as st
from core import suggest_model, ensure_model_exists, load_vector_store, build_vector_store, ollama_chat
from ingest import nomnom

st.set_page_config(page_title="AppBoi", layout="wide")
#st.title("AI Assistant - Hybrid (RAG & Pure Chat)")

@st.cache_resource
def get_default_model():
    return suggest_model()

# Model Selection
with st.sidebar:
    st.header("⚙️ Settings")

    default_model = get_default_model()
    #print("Got default Model")
    model_name = st.text_input("Ollama Model", default_model)
    
    if "prepared" not in st.session_state:
        st.session_state.prepared = False
    
    if not st.session_state.prepared:
        if st.button("Prepare Model & Data", key="prepare_btn"):
            with st.spinner("Preparing model and data... This may take a while."):
                ensure_model_exists(model_name)
                nomnom()
                build_vector_store()
            st.session_state.prepared = True
            st.success("✅ Model and data ready!")
    else:
        st.success("✅ Model and data already prepared.")

    # mode selection
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "RAG"
    st.session_state.active_mode = st.radio("Mode", ["RAG", "Pure Chat"], index=0 if st.session_state.active_mode == "RAG" else 1)

# Initialize Session State for histories
if "rag_history" not in st.session_state:
    st.session_state.rag_history = [{"role": "system", "content": "You are a helpful AI assistant using PDF context to answer questions."}]
if "pure_history" not in st.session_state:
    st.session_state.pure_history = [{"role": "system", "sontent": "Your are a helpful AI assistant using general knowledge to answer questions"}]

# Chat Display
st.title("💬 AI Assistant")

# Show history for active mode
if st.session_state.active_mode == "RAG":
    history = st.session_state.rag_history[1:] # skip system message
else:
    history = st.session_state.pure_history[1:]

for msg in history:
    if msg["role"] == "user":
        st.markdown(f"<div style='background-color:#606060; padding:10px; border-radius:10px; margin-bottom:5px; max-width:80%; align-self:flex-end'><b>You:<\b> {msg['content']}</div>", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"<div style='background-color:#666666; padding:10px; border-radius:10px; margin-bottom:5px; max-width:80%'><b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Chat Input
query = st.chat_input("Type your message and press Enter...")

if query:
    if st.session_state.active_mode == "RAG":
        vectorstore = load_vector_store()
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        user_input = f"Answer the following questions based on this context:\n\n{context}\n\nQuestion: {query}"
        st.session_state.rag_history.append({"role": "user", "content": query})
        reply = ollama_chat(model_name, st.session_state.rag_history[:-1] + [{"role": "user", "content": user_input}])
        st.session_state.rag_history.append({"role": "assistant", "content": reply})
    else:
        st.session_state.pure_history.append({"role": "user", "content": query})
        reply = ollama_chat(model_name, st.session_state.pure_history)
        st.session_state.pure_history.append({"role": "assistant", "content": reply})

