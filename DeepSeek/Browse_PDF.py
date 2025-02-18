# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import InMemoryVectorStore
# from langchain.memory import ConversationBufferMemory  # ✅ FIXED IMPORT
# from langchain.chains import ConversationalRetrievalChain
# from langchain_ollama import OllamaLLM
# from htmlTemplates1 import css, bot_template, user_template


# EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
# LANGUAGE_MODEL = OllamaLLM(model="llama3.1")
# DOCUMENT_VECTOR_DB = InMemoryVectorStore(embedding=EMBEDDING_MODEL)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""  # Handle None values
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
#     )
#     return text_splitter.split_text(text)

# def get_vectorstore(text_chunks):
#     DOCUMENT_VECTOR_DB.add_texts(texts=text_chunks)  # Adds texts in place
#     return DOCUMENT_VECTOR_DB  # Ensure we return the vector store object

# def get_conversation_chain(vectorstore):
#     if "memory" not in st.session_state:
#         st.session_state.memory = ConversationBufferMemory(
#             memory_key="chat_history", return_messages=True
#         )
    
#     return ConversationalRetrievalChain.from_llm(
#         llm=LANGUAGE_MODEL,
#         retriever=vectorstore.as_retriever(),  
#         memory=st.session_state.memory
#     )

# def handle_userinput(user_question):
#     if not st.session_state.conversation:
#         st.warning("Please upload and process a document first.")
#         return

#     response = st.session_state.conversation({"question": user_question})
#     st.session_state.chat_history = response.get("chat_history", [])

#     for i, message in enumerate(st.session_state.chat_history):
#         msg_template = user_template if i % 2 == 0 else bot_template
#         st.write(msg_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# def main():
#     st.write(css, unsafe_allow_html=True)

#     # Hide Streamlit branding
#     st.markdown(
#         """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""",
#         unsafe_allow_html=True,
#     )

#     # Text input fix
#     st.markdown(
#         """<style>.stTextInput {margin-bottom: 1rem;}</style>""",
#         unsafe_allow_html=True,
#     )

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     user_question = st.text_input("About EXIM Compliance:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.header("EXIM Chat :books:")
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 st.session_state.chat_history = []  # Reset chat history
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 vectorstore = get_vectorstore(text_chunks)
#                 st.session_state.conversation = get_conversation_chain(vectorstore)

#         st.markdown("[Made by: Alluvium IoT Solutions Pvt Ltd](https://www.alluvium.in/)")

# if __name__ == "__main__":
#     main()





import streamlit as st
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document



st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
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

EMBEDDING_MODEL = OllamaEmbeddings(model="all-minilm",num_gpu = 1, num_thread=12)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.2", num_gpu = 1)

def load_excel_documents(file):
    df = pd.read_excel(file)
    return df.to_string()

def chunk_documents(raw_text):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_text(raw_text)

def index_documents(document_chunks):
    doc_objects = [Document(page_content=chunk) for chunk in document_chunks]
    DOCUMENT_VECTOR_DB.add_documents(doc_objects)


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    # Extract text correctly from Document objects
    context_text = "\n\n".join([doc.page_content for doc in context_documents])  # ✅ Fix

    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

st.title("📊 ExcelMind AI")
st.markdown("### Your Intelligent Excel Assistant")
st.markdown("---")

# File Upload Section
uploaded_excel = st.file_uploader(
    "Upload Data File (Excel)",
    type=["xlsx", "xls"],
    help="Select an Excel file for analysis",
    accept_multiple_files=False )

if uploaded_excel:
    raw_text = load_excel_documents(uploaded_excel)
    processed_chunks = chunk_documents(raw_text)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)




