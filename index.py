import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores.qdrant import Qdrant

import config


def create_index(file_path: str) -> str:
    try:
        words = file_path.split('/')[:-1]
        document_dir = '/'.join(words)
        loader = DirectoryLoader(
            document_dir,
            glob='**/*.pdf',
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=32
        )
        texts = text_splitter.split_documents(documents)
        Qdrant.from_documents(
            texts,
            config.embedding_function,
            collection_name=config.QDRANT_COLLECTION_NAME,
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        os.unlink(file_path)
        return 'Documents uploaded and index created successfully. You can chat now.'
    except Exception as e:
        try:
            os.unlink(file_path)
        except:
            pass
        return e


def clear_index() -> tuple:
    config.qdrant_client.delete_collection(
        collection_name=config.COLLECTION_NAME)
    return 'Document cleared.', None
