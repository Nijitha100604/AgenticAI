from src.data_loader import load_all_documents
from src.chunk_and_embedding import EmbeddingPipeline
import os
import faiss
import numpy as np 
from sentence_transformers import SentenceTransformer
from typing import List, Any
import pickle


class VectorStore:

    # Constructor to initalize directory, embedding model and chunk details
    def __init__(self, persist_dir: str = "faiss_vectorstore", model_name: str = "all-MiniLM-L6-v2", chunk_size:int = 1000, chunk_overlap:int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok = True)
        self.embedding_model = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = None
        self.metadata = []
        self.model = SentenceTransformer(self.embedding_model)
        print(f"Loaded embedding model: {model_name}")


    # building the vector store with embeddings 
    def build_from_documents(self, documents: List[Any]):
        print(f"Building vector store from {len(documents)} raw documents...")
        embed_pipe = EmbeddingPipeline()
        chunks = embed_pipe.chunk_documents(documents)
        embeddings = embed_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"Vector store built and saved to {self.persist_dir}")


    # adding embeddings to the vector
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"Added {embeddings.shape[0]} vectors to Faiss index")


    # save the embeddings 
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved Faiss index and metadata to {self.persist_dir}")

    
    # load the vector store
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded Faiss index and metadata from {self.persist_dir}")


    # search for similar vectors
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results


    # query the vector store with a query text
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)


if __name__ == "__main__":
    docs = load_all_documents("data")
    store = VectorStore("faiss_vectorstore")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What are the categories of leave ?", top_k = 3))