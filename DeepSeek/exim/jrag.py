import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from htmlTemplates1 import css, bot_template, user_template

EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
LANGUAGE_MODEL = OllamaLLM(model="llama3.1")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(embedding=EMBEDDING_MODEL)

def get_json_text(json_files):
    text = ""
    for json_file in json_files:
        content = json.load(json_file)
        if isinstance(content, dict):
            text += " ".join(map(str, content.values()))  # Extract values from JSON dict
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text += " ".join(map(str, item.values())) + "\n"
                else:
                    text += str(item) + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ""], chunk_size=1000, chunk_overlap=100, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    batch_size = 10  # Process in smaller batches for efficiency
    for i in range(0, len(text_chunks), batch_size):
        DOCUMENT_VECTOR_DB.add_texts(texts=text_chunks[i:i + batch_size])
    return DOCUMENT_VECTOR_DB

def get_conversation_chain(vectorstore):
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    
    return ConversationalRetrievalChain.from_llm(
        llm=LANGUAGE_MODEL,
        retriever=vectorstore.as_retriever(),  
        memory=st.session_state.memory
    )

def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload and process a JSON file first.")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response.get("chat_history", [])

    for i, message in enumerate(st.session_state.chat_history):
        msg_template = user_template if i % 2 == 0 else bot_template
        st.write(msg_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.write(css, unsafe_allow_html=True)

    st.markdown(
        """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<style>.stTextInput {margin-bottom: 1rem;}</style>""",
        unsafe_allow_html=True,
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask about EXIM Compliance:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.header("EXIM Chat :books:")
        st.subheader("Your JSON files")
        json_files = st.file_uploader("Upload your JSON files here", accept_multiple_files=True, type=["json"])

        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.chat_history = []
                raw_text = get_json_text(json_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.markdown("[Made by: Alluvium IoT Solutions Pvt Ltd](https://www.alluvium.in/)")

if __name__ == "__main__":
    main()
