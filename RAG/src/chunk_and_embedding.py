from src.dataLoader import load_all_documents
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np 
from typing import List, Any

class EmbeddingPipeline:

    # Constructor to initialize the embedding model, chunks size
    def __init__(self, model_name:str = "all-MiniLM-L6-V2", chunk_size: int = 1000, chunk_overlap:int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    
    # chunking the documents 
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators  = ["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"Splitted {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    
    # chunks embedding
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
