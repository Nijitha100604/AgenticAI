import os
from dotenv import load_dotenv
from src.vectorstore_faiss import VectorStore
from src.dataLoader import load_all_documents
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:

    # constructor to initailize the vector store, embedding model, llm model
    def __init__(self, persist_dir: str = "faiss_vectorstore", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = VectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):  
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"Groq LLM initialized: {llm_model}")

    
    # search for similar vectors and summarize the content
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content  


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the categories of leave?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)