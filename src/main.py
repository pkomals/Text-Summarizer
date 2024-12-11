import os
import streamlit as st
import pickle
import time
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma
from unstructured.partition.auto import partition
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from dotenv import load_dotenv
load_dotenv() 

st.title("Document Reader")

st.sidebar.title("Document URL")
urls=[]

for i in range(2):
    url=st.sidebar.text_input(f"url- {i+1}")
    urls.append(url)

process_url_clicked= st.sidebar.button("Process URL")

file_path='vectorindex_HF.pkl'
repo_id="mistralai/Mistral-7B-Instruct-v0.3"

llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7)

main_placeholder= st.empty()

if process_url_clicked:
    #Data Loading
    loader=UnstructuredURLLoader(urls=urls,verify_ssl=False)
    main_placeholder.text("Data Loading Started.......")
    data= loader.load()
    if not data:
        main_placeholder.text("No data loaded. Please check the URLs.")
        

    # Split data into chunks
    main_placeholder.text("Text Splitter Started.......")
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size= 1000
    )

    docs=text_splitter.split_documents(data)

    #Embedding
    main_placeholder.text("Embedding Vector Started.......")
    embedding=HuggingFaceEmbeddings()
    vectorindex_HF=FAISS.from_documents(docs,embedding) 
    time.sleep(2)
    
    with open(file_path,"wb") as f:
        pickle.dump(vectorindex_HF,f)

   
query= main_placeholder.text_input("Question: ")
    
if query:
        if os.path.exists(file_path):
            with open(file_path,"rb") as f:
                vectorstore= pickle.load(f)
                chain= RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                st.write(result["answer"])

                #Display the source
                sources=result.get("sources","")
                if sources:
                     st.subheader("Sources:")
                     source_list=sources.split("\n")
                     for source in source_list:
                          st.write(source)