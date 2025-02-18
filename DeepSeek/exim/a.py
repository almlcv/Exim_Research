import streamlit as st
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Apply Streamlit styles
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Initialize models and vector store in session state for persistence
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

if "doc_vector_db" not in st.session_state:
    st.session_state.doc_vector_db = InMemoryVectorStore(st.session_state.embedding_model)

if "language_model" not in st.session_state:
    st.session_state.language_model = OllamaLLM(model="llama3.1")

DOCUMENT_VECTOR_DB = st.session_state.doc_vector_db
LANGUAGE_MODEL = st.session_state.language_model

# Function to load and parse JSON file
def load_json(file):
    try:
        return json.load(file)
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format. Please upload a valid JSON file.")
        return None

# Function to flatten and extract all JSON data dynamically
def extract_full_json(json_data):
    return json.dumps(json_data, indent=2)  # Convert full JSON into text format

# Function to chunk text for better retrieval
def chunk_documents(raw_text):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True)
    
    return text_processor.split_text(raw_text)

# Function to index full JSON document dynamically
def index_documents(json_data):
    json_text = extract_full_json(json_data)
    chunks = chunk_documents(json_text)
    DOCUMENT_VECTOR_DB.add_texts([doc for doc in chunks])

# Function to perform semantic search in vector store
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Function to generate AI response based on retrieved documents
def generate_answer(user_query, context_documents):
    if not context_documents:
        return "I couldn't find relevant information for your query."

    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Streamlit UI
st.title("📘 AI-Powered JSON Query")
st.markdown("### Retrieve dynamic information from uploaded JSON")
st.markdown("---")

uploaded_json = st.file_uploader("Upload JSON file", type="json", help="Select a JSON file containing data")

if uploaded_json:
    json_data = load_json(uploaded_json)
    if json_data:
        # Index the full JSON document
        index_documents(json_data)
        
        st.success(f"✅ Indexed document successfully. Ready for queries.")
        
        user_input = st.chat_input("Ask any question related to the document...", key="document_query")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Searching for relevant information..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)
