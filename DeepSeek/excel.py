# import streamlit as st
# import pandas as pd
# import gc
# from typing import List, Dict, Any
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import InMemoryVectorStore
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_ollama import OllamaLLM
# from htmlTemplates1 import css, bot_template, user_template

# st.set_page_config(
#     page_title="EXIM Chat",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize models with configurable parameters
# @st.cache_resource
# def init_models():
#     return {
#         "embedding": OllamaEmbeddings(model="nomic-embed-text", num_gpu=1, num_thread=12),
#         "language": OllamaLLM(model="llama3.2", num_gpu=1)
#     }

# MODELS = init_models()

# def process_excel_content(df: pd.DataFrame, sheet_name: str) -> str:
#     """
#     Process DataFrame content into structured text that preserves relationships and context.
#     """
#     text_content = []
    
#     # Add sheet name context
#     text_content.append(f"Sheet: {sheet_name}")
    
#     # Process column headers with their data types
#     headers = df.columns.tolist()
#     text_content.append("Columns: " + ", ".join(f"{col} ({df[col].dtype})" for col in headers))
    
#     # Group related information for each unique entity
#     entity_columns = ['JOB NO AND DATE', 'IMPORTER', 'SUPPLIER/ EXPORTER', 'INVOICE NUMBER AND DATE','INVOICE VALUE AND UNIT PRICE','BL NUMBER AND DATE','COMMODITY','NET WEIGHT', 'PORT', 'ARRIVAL DATE', 'FREE TIME', 'DETENTION FROM', 'SHIPPING LINE', 'CONTAINER NUM & SIZE', 'NUMBER OF CONTAINERS', 'BE NUMBER AND DATE','REMARKS', 'DETAILED STATUS']
#     available_entity_cols = [col for col in entity_columns if col in headers]
    
#     if available_entity_cols:
#         for entity_col in available_entity_cols:
#             unique_entities = df[entity_col].dropna().unique()
#             for entity in unique_entities:
#                 entity_data = df[df[entity_col] == entity]
                
#                 # Create a detailed context block for each entity
#                 entity_info = [f"\n{entity_col}: {entity}"]
                
#                 # Add all related information
#                 for col in headers:
#                     if col != entity_col:
#                         values = entity_data[col].dropna().unique()
#                         if len(values) > 0:
#                             formatted_values = '; '.join(str(v) for v in values)
#                             entity_info.append(f"{col}: {formatted_values}")
                
#                 text_content.append('\n'.join(entity_info))
#     else:
#         # If no entity columns found, process row by row with context
#         for idx, row in df.iterrows():
#             row_text = [f"\nRecord {idx + 1}:"]
#             for col in headers:
#                 if pd.notna(row[col]):
#                     row_text.append(f"{col}: {row[col]}")
#             text_content.append('\n'.join(row_text))
    
#     return '\n\n'.join(text_content)

# @st.cache_data(max_entries=1)
# def get_excel_text(excel_files) -> str:
#     """
#     Process Excel files with improved context preservation and progress tracking.
#     """
#     progress_bar = st.progress(0)
#     text_chunks = []
#     total_files = len(excel_files)
    
#     for idx, excel_file in enumerate(excel_files):
#         try:
#             # Read all sheets in the Excel file
#             file_name = excel_file.name
#             text_chunks.append(f"\nFile: {file_name}")
            
#             xls = pd.ExcelFile(excel_file)
#             for sheet_name in xls.sheet_names:
#                 df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
#                 # Clean and preprocess the DataFrame
#                 df = df.replace({'\n': ' ', '\r': ' '}, regex=True)  # Remove newlines in cells
#                 df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)  # Strip whitespace
                
#                 # Process the sheet's content
#                 sheet_content = process_excel_content(df, sheet_name)
#                 text_chunks.append(sheet_content)
                
#         except Exception as e:
#             st.error(f"Error processing file {file_name}: {e}")
        
#         progress = min(1.0, (idx + 1) / total_files)
#         progress_bar.progress(progress)
    
#     return "\n\n".join(text_chunks)

# def get_text_chunks(text: str, chunk_size: int = 500) -> List[str]:
#     """
#     Split text into optimized chunks while preserving context.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". ", " ", ""],
#         chunk_size=chunk_size,
#         chunk_overlap=100,  # Increased overlap to maintain context
#         length_function=len
#     )
#     return splitter.split_text(text)

# @st.cache_resource
# def create_embeddings(texts: List[str], batch_size: int = 200) -> InMemoryVectorStore:
#     vectorstore = InMemoryVectorStore(embedding=MODELS["embedding"])
#     progress_bar = st.progress(0)
    
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         vectorstore.add_texts(texts=batch)
#         progress = min(1.0, (i + batch_size) / len(texts))
#         progress_bar.progress(progress)
    
#     return vectorstore

# def get_conversation_chain(vectorstore: InMemoryVectorStore):
#     """Initialize the conversational retrieval chain with improved context handling."""
#     if "memory" not in st.session_state:
#         st.session_state.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             max_token_limit=2000
#         )
    
#     return ConversationalRetrievalChain.from_llm(
#         llm=MODELS["language"],
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Increased k for better context
#         memory=st.session_state.memory,
#         max_tokens_limit=4000
#     )

#     conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     response_chain = conversation_prompt | LANGUAGE_MODEL
#     return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# def handle_userinput(user_question: str):
#     """Handle user input by retrieving context and generating an answer."""
#     if not st.session_state.conversation:
#         st.warning("Please upload and process an Excel file first.")
#         return
    
#     try:
#         response = st.session_state.conversation.invoke({"question": user_question})
#         st.session_state.chat_history = response.get("chat_history", [])
        
#         for i, message in enumerate(st.session_state.chat_history):
#             msg_template = user_template if i % 2 == 0 else bot_template
#             st.write(msg_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error processing question: {str(e)}")

# def clear_memory():
#     """Clear session memory and cached data."""
#     if "memory" in st.session_state:
#         del st.session_state.memory
#     if "conversation" in st.session_state:
#         del st.session_state.conversation
#     if "chat_history" in st.session_state:
#         del st.session_state.chat_history
    
#     st.cache_data.clear()
#     gc.collect()

# def main():
#     st.write(css, unsafe_allow_html=True)
    
#     # Hide Streamlit default UI elements
#     st.markdown("""
#         <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             .stTextInput {margin-bottom: 1rem;}
#         </style>
#     """, unsafe_allow_html=True)
    
#     # Initialize session state variables
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Main chat interface
#     user_question = st.text_input("Ask about EXIM Compliance:")
#     if user_question:
#         handle_userinput(user_question)
    
#     # Sidebar with configuration and file upload options
#     with st.sidebar:
#         st.title("EXIM Compliance Chat :books:")
#         st.header("Settings")
        
#         chunk_size = st.slider("Text Chunk Size", 200, 1000, 500)
#         batch_size = st.slider("Processing Batch Size", 10, 200, 50)
        
#         st.subheader("File Upload")
#         excel_files = st.file_uploader(
#             "Upload Excel files",
#             accept_multiple_files=True,
#             type=["xlsx", "xls"]
#         )
        
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Process Files"):
#                 try:
#                     with st.spinner("Processing..."):
#                         clear_memory()
#                         st.info("Reading Excel files...")
#                         raw_text = get_excel_text(excel_files)
#                         st.info("Creating text chunks...")
#                         text_chunks = get_text_chunks(raw_text, chunk_size)
#                         st.info("Creating embeddings...")
#                         vectorstore = create_embeddings(text_chunks, batch_size)
#                         st.info("Setting up chat...")
#                         st.session_state.conversation = get_conversation_chain(vectorstore)
#                         st.success("Ready to chat!")
#                 except Exception as e:
#                     st.error(f"Error during processing: {str(e)}")
        
#         with col2:
#             if st.button("Clear Memory"):
#                 clear_memory()
#                 st.success("Memory cleared!")
        
#         st.markdown("---")
#         st.markdown("[Made by: Alluvium IoT Solutions Pvt Ltd](https://www.alluvium.in/)")

# if __name__ == "__main__":
#     main()












import pandas as pd
import gc
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import (create_history_aware_retriever, create_retrieval_chain,)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import PromptTemplate
template = ""

def init_models():
    return {
        "embedding": OllamaEmbeddings(model="nomic-embed-text", num_gpu=1, num_thread=12),
        "language": OllamaLLM(model="llama3.2", num_gpu=1)  # Updated model name
 }

MODELS = init_models()




def process_excel_content(df: pd.DataFrame, sheet_name: str) -> str:
    text_content = [f"Sheet: {sheet_name}"]
    headers = df.columns.tolist()
    text_content.append("Columns: " + ", ".join(f"{col} ({df[col].dtype})" for col in headers))
    
    for idx, row in df.iterrows():
        row_text = [f"\nRecord {idx + 1}:"]
        for col in headers:
            if pd.notna(row[col]):
                row_text.append(f"{col}: {row[col]}")
        text_content.append('\n'.join(row_text))
    
    return '\n\n'.join(text_content)

def get_excel_text(file_path: str) -> str:
    text_chunks = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df = df.replace({'\n': ' ', '\r': ' '}, regex=True)
            df = df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
            text_chunks.append(process_excel_content(df, sheet_name))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return '\n\n'.join(text_chunks)

def get_text_chunks(text: str, chunk_size: int = 500) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

def create_embeddings(texts: List[str], batch_size: int = 200) -> InMemoryVectorStore:
    vectorstore = InMemoryVectorStore(embedding=MODELS["embedding"])
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectorstore.add_texts(texts=batch)
    return vectorstore

def get_conversation_chain(vectorstore: InMemoryVectorStore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  
    return ConversationalRetrievalChain.from_llm(
        llm=MODELS["language"],
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        question_generator=question_generator_chain
    )


def chat_interface():
    file_path = "DS.xlsx"
    raw_text = get_excel_text(file_path)  # Extracts text from Excel
    text_chunks = get_text_chunks(raw_text)  # Splits text
    vectorstore = create_embeddings(text_chunks)  # Embeds text
    conversation = get_conversation_chain(vectorstore)  # Initializes chat system
    
    print("\nChat ready! Type 'exit' to quit.\n")

    chat_history = []  # Initialize chat history
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        
        # ✅ Include chat_history when invoking
        response = conversation.invoke({
            "question": user_input,
            "chat_history": chat_history  # Add history
        })
        
        chat_history = response.get("chat_history", [])  
        
        # ✅ Iterate over chat history properly
        for i, message in enumerate(chat_history[-2:]):  
            if i % 2 == 0:
                print(f"You: {message.content}")
            else:
                print(f"Bot: {message.content}")
    
    print("\nChat session ended.")


if __name__ == "__main__":
    chat_interface()






