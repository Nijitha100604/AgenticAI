from src.dataLoader import load_all_documents
from src.chunk_and_embedding import EmbeddingPipeline
from src.vectorstore_faiss import VectorStore
from src.search_with_llm import RAGSearch

def main():

    # documents = load_all_documents("data")
    # embedding_pipe = EmbeddingPipeline()
    # chunks = embedding_pipe.chunk_documents(documents)
    # embeddings = embedding_pipe.embed_chunks(chunks)
    # print("Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
    
    # documents = load_all_documents("data")
    # store = VectorStore("faiss_vectorstore")
    # store.build_from_documents(documents)
    # store.load()
    # print(store.query("What are the categories of leave ?", top_k = 3))

    rag_search = RAGSearch()
    query = "What are the categories of leave?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

if __name__ == "__main__":
    main()