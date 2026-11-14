"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.core.config import settings

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.GEMINI_EMBEDDING_MODEL,
    google_api_key=settings.GOOGLE_API_KEY
)

# Global variable to store the FAISS vectorstore instance
_faiss_vectorstore = None
INDEX_DIR = "faiss_index"


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in FAISS vector database.
    """
    global _faiss_vectorstore

    try:
        is_dummy = False
        if _faiss_vectorstore is not None:
            try:
                if hasattr(_faiss_vectorstore, "docstore") and len(_faiss_vectorstore.docstore._dict) <= 1:
                    first_doc = next(iter(_faiss_vectorstore.docstore._dict.values()), None)
                    if first_doc and first_doc.metadata.get("source") == "initialization":
                        is_dummy = True
            except Exception:
                pass

        if _faiss_vectorstore is None or is_dummy:
            _faiss_vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
        else:
            _faiss_vectorstore.add_documents(chunks)

        try:
            _faiss_vectorstore.save_local(INDEX_DIR)
            print(f"FAISS vector store saved to {INDEX_DIR}")
        except Exception as e:
            print(f"Warning: Could not save FAISS to disk: {e}")

        print("FAISS vector store initialized/updated with documents")
        print(f"Added {len(chunks)} document chunks")
        return True

    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        return False


def get_retriever():
    """
    Get a retriever tool connected to the FAISS vector store.
    """
    global _faiss_vectorstore

    try:
        if _faiss_vectorstore is None and os.path.exists(INDEX_DIR):
            try:
                _faiss_vectorstore = FAISS.load_local(
                    INDEX_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded FAISS vectorstore from {INDEX_DIR}")
            except Exception as e:
                print(f"Could not load FAISS from disk: {e}")

        if _faiss_vectorstore is not None:
            retriever = _faiss_vectorstore.as_retriever()
            print("Using existing FAISS vectorstore with uploaded documents")

        else:
            print("No documents uploaded yet, creating dummy vectorstore")

            from langchain_core.documents import Document as LangChainDocument

            dummy_doc = LangChainDocument(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"}
            )

            _faiss_vectorstore = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings
            )

            retriever = _faiss_vectorstore.as_retriever()

        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read().strip()
        else:
            description = "uploaded documents"

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool to answer questions about: {description}\n"
            "Do not use this tool for anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)
