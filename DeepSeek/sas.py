import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import UnstructuredExcelLoader
import os
import tempfile
import torch

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

def load_qa_system(file_path):
    loader = UnstructuredExcelLoader(file_path)
    documents = loader.load()
    print("This is the documents: \n\n",documents)
    
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  # Adjust based on dataset
        chunk_overlap=30,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    print("This is the splitted docuemnts: \n\n", docs)
    
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    
    FAISS_PATH = f"{file_path}.faiss"
    
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(FAISS_PATH, embeddings)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(FAISS_PATH)

    retriever = vectorstore.as_retriever()
    
    llm = OllamaLLM(model="llama3.2", num_gpu=1)
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return qa



def main():
    st.set_page_config(page_title="Excel QA System", layout="wide")
    
    st.title("📊 Excel Question Answering System")
    st.write("Upload your Excel file and ask questions about its contents!")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())
        
        try:
            qa_system = load_qa_system(file_path)
            st.success("File loaded successfully! You can now ask questions.")
            
            with st.form("question_form"):
                question = st.text_input("Ask a question about your Excel file:")
                submit_button = st.form_submit_button("Get Answer")
                
                if submit_button and question:
                    with st.spinner("Finding answer..."):
                        try:
                            answer = qa_system.run(question)
                            st.write("### Answer:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error getting answer: {str(e)}")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
        finally:
            os.remove(file_path)
    
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Upload your Excel file
        2. Wait for processing
        3. Type your question
        4. Click 'Get Answer'
        5. View your history
        """)

if __name__ == "__main__":
    main()


