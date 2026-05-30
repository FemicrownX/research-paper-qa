from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def retrieve(vectorstore, query, k=10):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "fetch_k": 50000
        }
    )
    results = retriever.invoke(query)
    return results