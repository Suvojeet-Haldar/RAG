from retriever import retriever
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
import os

from dotenv import load_dotenv
load_dotenv()

groq_api_key =os.getenv('GROQ_API_KEY')

st.title("US Census RAG QA Bot")

st.markdown('Using <a href="https://build.nvidia.com/nvidia/nv-embed-v1" target="_blank">nvidia/nv-embed-v1</a> for embedding, <a href="https://python.langchain.com/v0.2/docs/integrations/retrievers/pinecone_hybrid_search/" target="_blank">Pinecone</a> for Hybrid Search Retrieval & Llama-3.1-70b-Versatile(<a href="https://groq.com/#" target="_blank">Groq</a>) as the llm.', unsafe_allow_html=True)

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

input_prompt= st.text_input(f"Enter Your Questions from the documents: https://drive.google.com/drive/folders/1AjuSJdE9ekD4dTkbKD58w5AUuxRWe052?usp=sharing")

# st.markdown('<a href="https://acrobat.adobe.com/id/urn:aaid:sc:AP:dd728725-7701-4af7-b4a1-bb35a20f6d4e" target="_blank">Document</a>', unsafe_allow_html=True)

if input_prompt:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    try:
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