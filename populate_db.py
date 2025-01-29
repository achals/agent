import os
from typing import List
from uuid import uuid4
import mimetypes

from tqdm import tqdm

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="personal_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Source directory location for embedding
SOURCE_DIRS = ["/Users/achal/Documents", "/Users/achal/Downloads"]

def get_text_files() -> List[str]:
    text_files = []
    for SOURCE_DIR in SOURCE_DIRS:
        for root, dirs, files in os.walk(SOURCE_DIR):
            for file in files:
                _path = os.path.join(root, file)
                mime = mimetypes.guess_type(_path)
                if mime[0] and "text" in mime[0]:
                    text_files.append(str(_path))
    return text_files

def get_pdf_files():
    for SOURCE_DIR in SOURCE_DIRS:
        loader = PyPDFDirectoryLoader(SOURCE_DIR)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vector_store.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

def populate_db(files: list[str]):
    for _path in tqdm(files):
        with open(_path, "r") as f:
            try:
                content = f.read()
                _id = str(uuid4())
                doc = Document(
                    id=str(uuid4()),
                    page_content=content,
                    metadata={"path": _path},
                )
                vector_store.add_documents(documents=[doc], ids=[_id])
            except:
                print(f"Error processing {_path}")
                continue


if __name__ == "__main__":
    _files = get_text_files()
    get_pdf_files()

    populate_db(_files)