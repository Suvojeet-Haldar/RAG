from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone,ServerlessSpec
import os

from dotenv import load_dotenv
load_dotenv()


# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")

pinecone_api_key=os.getenv('PINECONE_API_KEY')

index_name="hybrid-search-us-census"
# Iniialize the Pinecone client
pc=Pinecone(api_key=pinecone_api_key)

# create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096, #dimensionality of dense model
        metric ="dotproduct", #sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index=pc.Index(index_name)

# vector embedding and sparse matrix
embeddings=NVIDIAEmbeddings(model="nvidia/nv-embed-v1")


loader = PyPDFDirectoryLoader("./us_census")
docs= loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents= text_splitter.split_documents(docs)
corpus = [doc.page_content for doc in final_documents]


bm25_encoder=BM25Encoder().default()
# tfidf values on these sentence
bm25_encoder.fit(corpus)

# store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever=PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

retriever.add_texts(corpus)

print("retriever ready for use.")