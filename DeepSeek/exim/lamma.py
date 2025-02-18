import streamlit as st
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

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

PROMPT_TEMPLATE =  """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""




EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.1")

def load_json(file):
    return json.load(file)

def extract_container_info(json_data, container_number):
    for job in json_data:
        if "container_nos" in job:
            for container in job["container_nos"]:
                if isinstance(container, dict) and container.get("container_number") == container_number:
                    return {
                        "job_no": job["job_no"],
                        "year": job["year"],
                        "awb_bl_no": job["awb_bl_no"],
                        "be_no": job["be_no"],
                        "importer": job["importer"],
                        "loading_port": job["loading_port"],
                        "port_of_reporting": job["port_of_reporting"],
                        "supplier_exporter": job["supplier_exporter"],
                        "consignment_type": job["consignment_type"],
                        "container_details": container  }
    return None

def chunk_documents(raw_text):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True)
    
    return text_processor.split_text(raw_text)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_texts(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

st.title("📘 Container Info AI")
st.markdown("### Retrieve container details from JSON")
st.markdown("---")

uploaded_json = st.file_uploader("Upload JSON file", type="json", help="Select a JSON file containing container data")

if uploaded_json:
    json_data = load_json(uploaded_json)
    container_number = st.text_input("Enter Container Number:")
            
    if container_number:
        container_info = extract_container_info(json_data, container_number)

        if container_info:
            st.success("✅ Container data found!")
            st.subheader("Container Details:")
            st.json(container_info)
            
            processed_chunks = chunk_documents(json.dumps(container_info, indent=2))
            index_documents(processed_chunks)
            
            user_input = st.chat_input("Enter your question about the container...", key="container_query")
            
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.spinner("Analyzing container data..."):
                    relevant_docs = find_related_documents(user_input)
                    ai_response = generate_answer(user_input, relevant_docs)
                    
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(ai_response)
        else:
            st.error("❌ No data found for the given container number.")








