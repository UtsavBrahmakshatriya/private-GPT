import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import huggingface_hub

from datetime import datetime

from htmltemplates import css, bot_template, user_template

def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    chroma_db = chroma.Chroma()
    embeddings = HuggingFaceInstructEmbeddings(model_name= "hkunlp/instructor-xl")
    vectorstore = chroma_db.from_texts(texts=text_chunks,embedding=embeddings,persist_directory="./data/vectordb/")
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = huggingface_hub.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i%2==0:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Private ChatGPT",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Private ChatGPT :lock:")
    container = st.container()
    user_question = st.text_input("Ask me anything")
    if st.button("Find Answer"):
        with st.spinner("Finding..."):
            handle_userinput(user_question)

    with st.sidebar:
        st.header("Admin Zone")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True,type="PDF")
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_text_from_pdf(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                start_time = datetime.now()
                vectorstore = get_vectorstore(text_chunks)
                end_time = datetime.now()
                print("time taken:",end_time-start_time)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
if __name__ == '__main__':
    main()