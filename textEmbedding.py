from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import chromadb

loader = PyPDFLoader("ConroyTaggartIMR.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(pages)

model = SentenceTransformer('all-MiniLM-L6-v2')

persist_directory = 'docs/chroma/' #it will create a docs/chroma each time running it
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

