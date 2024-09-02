import streamlit as st
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()
import getpass
import os

# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")

groq_api_key =os.getenv('GROQ_API_KEY')

st.title("US Census RAG QA Bot")

st.markdown('Using <a href="https://build.nvidia.com/nvidia/nv-embed-v1" target="_blank">nvidia/nv-embed-v1</a> for embedding, <a href="https://objectbox.io/" target="_blank">Objectbox</a> as VectorDB & Llama-3.1-70b-Versatile(<a href="https://groq.com/#" target="_blank">Groq</a>) as the llm.', unsafe_allow_html=True)

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

# Vector Embedding and Objectbox Vectorstore db

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ## Data Ingestion
        
        st.session_state.docs= st.session_state.loader.load() ## Document Loading
        
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ## Chunk Creation
        
        st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs) ## Splitting

        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=4096) ## Vector HuggingFaceBgeEmbeddings
    



if st.button("Document(s) Embedding"):
    start=time.process_time()
    vector_embedding()
    st.markdown(":green[Vector Store DB Is Ready], you can now ask questions")
    st.write("Embedding time :", time.process_time()-start)

input_prompt= st.text_input(f"Enter Your Questions from the document: https://drive.google.com/file/d/1GOb4COSLf-dBp0Rmu_b9goPSgrhBg2GQ/view?usp=sharing")

# st.markdown('<a href="https://acrobat.adobe.com/id/urn:aaid:sc:AP:dd728725-7701-4af7-b4a1-bb35a20f6d4e" target="_blank">Document</a>', unsafe_allow_html=True)





if input_prompt:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    try:
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
    except AttributeError:
        # Code to handle the exception
        st.markdown(":red[Please first embed document(s) before asking questions.]")