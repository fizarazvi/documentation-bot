from dotenv import load_dotenv
import os

load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():

    doc_path = "langchain-docs/api.python.langchain.com/en/latest"
    if not os.path.exists(doc_path):
        print(f"Path does not exist: {doc_path}")
        return

    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding="ISO-8859-1")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )

    print("****loading to vectorStore****")



if __name__ == "__main__":
    ingest_docs()