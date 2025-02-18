import streamlit as st

st.set_page_config(
    page_title="EXIM Chat",
    page_icon="📚",
    layout="wide"
)

import json
import gc
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from htmlTemplates1 import css, bot_template, user_template

# Initialize models with configurable parameters
@st.cache_resource
def init_models():
    return {
        # Ensure you have pulled the model "nomic-embed-text" or replace with a valid one.
        "embedding": OllamaEmbeddings(model="nomic-embed-text", num_gpu=1, num_thread=12),
        "language": OllamaLLM(model="llama3.2", num_gpu=1)
    }

MODELS = init_models()

def process_json_content(content: Any) -> str:
    """Recursively process JSON content into plain text."""
    if isinstance(content, dict):
        return " ".join(str(v) for v in content.values() if v is not None)
    elif isinstance(content, list):
        return "\n".join(process_json_content(item) for item in content)
    return str(content)

@st.cache_data(max_entries=1)
def get_json_text(json_files) -> str:
    """
    Process JSON files with progress tracking.
    Loads the entire JSON file at once (efficient for a 30 MB file).
    """
    progress_bar = st.progress(0)
    text_chunks = []
    total_files = len(json_files)
    
    for idx, json_file in enumerate(json_files):
        try:
            data = json.load(json_file)
            text_chunks.append(process_json_content(data))
        except Exception as e:
            st.error(f"Error processing file: {e}")
        progress = min(1.0, (idx + 1) / total_files)
        progress_bar.progress(progress)
    
    return "\n".join(text_chunks)

def get_text_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into optimized chunks using a recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)


@st.cache_resource
def create_embeddings(texts: List[str], batch_size: int = 200) -> InMemoryVectorStore:
    vectorstore = InMemoryVectorStore(embedding=MODELS["embedding"])
    progress_bar = st.progress(0)
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectorstore.add_texts(texts=batch)
        # Update progress bar only every 5 batches or at the final batch
        if ((i // batch_size) % 5 == 0) or (i + batch_size >= len(texts)):
            progress = min(1.0, (i + batch_size) / len(texts))
            progress_bar.progress(progress)
    
    progress_bar.progress(1.0)
    return vectorstore


def get_conversation_chain(vectorstore: InMemoryVectorStore):
    """
    Initialize the conversational retrieval chain using a conversation memory.
    This chain uses the vector store to retrieve relevant context.
    """
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000  # Limit memory size
        )
    
    return ConversationalRetrievalChain.from_llm(
        llm=MODELS["language"],
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        max_tokens_limit=4000  # Limit response size
    )

def handle_userinput(user_question: str):
    """Handle user input by retrieving context and generating an answer."""
    if not st.session_state.conversation:
        st.warning("Please upload and process a JSON file first.")
        return
    
    try:
        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response.get("chat_history", [])
        
        for i, message in enumerate(st.session_state.chat_history):
            msg_template = user_template if i % 2 == 0 else bot_template
            st.write(msg_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def clear_memory():
    """Clear session memory and cached data."""
    if "memory" in st.session_state:
        del st.session_state.memory
    if "conversation" in st.session_state:
        del st.session_state.conversation
    if "chat_history" in st.session_state:
        del st.session_state.chat_history
    
    st.cache_data.clear()
    gc.collect()

def main():
    st.write(css, unsafe_allow_html=True)
    
    # Hide Streamlit default UI elements
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stTextInput {margin-bottom: 1rem;}
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Main chat interface
    user_question = st.text_input("Ask about EXIM Compliance:")
    if user_question:
        handle_userinput(user_question)
    
    # Sidebar with configuration and file upload options
    with st.sidebar:
        st.title("EXIM Compliance Chat :books:")
        st.header("Settings")
        
        chunk_size = st.slider("Text Chunk Size", 200, 1000, 500)
        batch_size = st.slider("Processing Batch Size", 10, 200, 50)
        
        st.subheader("File Upload")
        json_files = st.file_uploader(
            "Upload JSON files",
            accept_multiple_files=True,
            type=["json"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Files"):
                try:
                    with st.spinner("Processing..."):
                        clear_memory()
                        st.info("Reading JSON files...")
                        raw_text = get_json_text(json_files)
                        st.info("Creating text chunks...")
                        text_chunks = get_text_chunks(raw_text, chunk_size)
                        st.info("Creating embeddings...")
                        vectorstore = create_embeddings(text_chunks, batch_size)
                        st.info("Setting up chat...")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Ready to chat!")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        
        with col2:
            if st.button("Clear Memory"):
                clear_memory()
                st.success("Memory cleared!")
        
        st.markdown("---")
        st.markdown("[Made by: Alluvium IoT Solutions Pvt Ltd](https://www.alluvium.in/)")
    
if __name__ == "__main__":
    main()



