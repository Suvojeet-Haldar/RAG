import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

groq_api_key =os.environ['GROQ_API_KEY']

st.title("HuggingFace 'BAAI/bge-large-en-v1.5' embedding model & Objectbox VectorDB With Groq(Llama-3.1-70b-Versatile) Demo")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama-3.1-70b-Versatile")

prompt_template=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

import torch
torch.cuda.set_device(0)  # Set the CUDA device to the first GPU (0)
# print(torch.cuda.is_available())  # Should print: True
# print(torch.cuda.current_device())  # Should print: 0 (the first GPU)


# Vector Embedding and Objectbox Vectorstore db

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True}
        )

        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        
        st.session_state.docs= st.session_state.loader.load() ## Document Loading
        
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ## Chunk Creation
        
        st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs) ## Splitting

        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=1024) ## Vector HuggingFaceBgeEmbeddings


input_prompt= st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if input_prompt:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({"input":input_prompt})

    st.write(response['answer'])
    st.write("Response time :", time.process_time()-start)


    # With a streamlit expander 
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")