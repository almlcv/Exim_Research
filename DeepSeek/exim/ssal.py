import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import UnstructuredExcelLoader
import os

def load_qa_system(file_path):
    # Load document
    loader = UnstructuredExcelLoader(file_path)
    documents = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    
    # Initialize embeddings
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    
    # Create and save vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Initialize retriever
    retriever = vectorstore.as_retriever()
    
    # Setup LLM
    llm = OllamaLLM(model="llama3.2", num_gpu=1)
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    return qa

def main():
    st.set_page_config(page_title="Excel QA System", layout="wide")
    
    # Header
    st.title("📊 Excel Question Answering System")
    st.write("Upload your Excel file and ask questions about its contents!")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp.xlsx", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Initialize QA system
            qa_system = load_qa_system("temp.xlsx")
            st.success("File loaded successfully! You can now ask questions.")
            
            # Question input
            with st.form("question_form"):
                question = st.text_input("Ask a question about your Excel file:")
                submit_button = st.form_submit_button("Get Answer")
                
                if submit_button and question:
                    with st.spinner("Finding answer..."):
                        try:
                            answer = qa_system.run(question)
                            
                            # Display answer in a nice format
                            st.write("### Answer:")
                            st.write(answer)
                            
                        except Exception as e:
                            st.error(f"Error getting answer: {str(e)}")
            
            # History section
            if 'history' not in st.session_state:
                st.session_state.history = []
                
            if submit_button and question:
                st.session_state.history.append((question, answer))
            
            # Display history
            if st.session_state.history:
                st.write("### Question History")
                for q, a in st.session_state.history:
                    with st.expander(f"Q: {q}"):
                        st.write("Answer:", a)
                        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists("temp.xlsx"):
                os.remove("temp.xlsx")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Upload your Excel file using the uploader
        2. Wait for the system to process the file
        3. Type your question in the text box
        4. Click 'Get Answer' to receive your response
        5. View your question history below
        """)
        


if __name__ == "__main__":
    main()