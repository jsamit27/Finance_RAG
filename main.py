import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS
from langchain_openai import OpenAI


from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")

st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
   url =  st.sidebar.text_input(f"URL :{i+1}")
   urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)

file_path ="faiss_index_store"

if process_url_clicked:
   # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("data loading started ....")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )

    main_placeholder.text("Text splitter started ....")

    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()

    vectorstore_openai= FAISS.from_documents(docs, embeddings)
    main_placeholder.text("embedding vector started building ....")
    time.sleep(2)

    vectorstore_openai.save_local(file_path)


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):

                        
        embeddings = OpenAIEmbeddings()
        vectorIndex = FAISS.load_local(
            folder_path=file_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # <-- explicitly allow it
        )
    else:
        vectorIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
        vectorIndex.save_local(file_path)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
    result = chain({"question":query}, return_only_outputs=True)

    st.header("Answer")
    st.write(result["answer"])
    
    # wanna display sources 
    sources=result.get("sources", "")

    if sources:
        st.subheader("sources: ")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)








