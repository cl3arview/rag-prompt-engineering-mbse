from pathlib import Path
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "ibm-granite/granite-embedding-30m-english"
CHROMA_DIR  = "chroma_db"
def load_and_split(pdf_path: Path, chunk_size: int = 750, chunk_overlap: int = 100):
    """
    Load a PDF with PyPDFium2Loader and split into overlapping text chunks.
    """
    pages = PyPDFium2Loader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(pages)

def build_vector_index(chunks, persist_dir: str = CHROMA_DIR):
    """
    Build and persist a Chroma vector index over the provided document chunks.
    """
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb
