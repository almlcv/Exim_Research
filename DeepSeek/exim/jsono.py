import os
import json
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Ensure the temp directory exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)  # ✅ Creates the temp directory if it does not exist

# Constants
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = None  # Will be initialized later

# Function to Convert JSON to Langchain Document Object
def json_to_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)  
    text_data = json.dumps(json_data, indent=4)  # Convert JSON to formatted text
    return Document(page_content=text_data)  # Return a Langchain Document object

# Function to Split Documents into Chunks
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Function to Index Documents in Vector Store
def index_documents(document_chunks):
    global DOCUMENT_VECTOR_DB
    DOCUMENT_VECTOR_DB = InMemoryVectorStore.from_documents(document_chunks, embedding=EMBEDDING_MODEL)

# Function to Retrieve Related Documents
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=3)  # Retrieve top 3 relevant docs

# Prompt Template for LLM Responses
PROMPT_TEMPLATE =  """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Function to Generate Answers from LLM
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# Streamlit UI
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Component
uploaded_file = st.file_uploader("Upload your document (JSON or PDF)", type=["json", "pdf"])

# Process Uploaded File
if uploaded_file:
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)  # ✅ Save file inside temp directory
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.type == "application/json":
        doc = json_to_document(file_path)  # Convert JSON to Document object
    else:
        loader = PDFPlumberLoader(file_path)  # Load PDF
        doc = loader.load()[0]  # Extract first document from loader output

    # Process and Index Document
    processed_chunks = chunk_documents([doc])
    index_documents(processed_chunks)
    st.success("✅ Document processed successfully! Ask your questions below.")

# User Query Input
user_input = st.chat_input("Enter your question about the document...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
