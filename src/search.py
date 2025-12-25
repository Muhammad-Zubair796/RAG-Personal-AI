import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2", llm_model="llama-3.1-8b-instant"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize vectorstore with first-run guard
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=self.embedding_model)
        self.vectorstore.get_or_build_vectorstore(data_dir="data/text_files")

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=llm_model,
            temperature=0.4,  # Slightly higher for natural responses
        )

    def search_and_summarize(self, query: str, top_k: int = 10) -> str:
        # Embed user query
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        results = self.vectorstore.search(query_embedding, top_k=top_k)

        texts = [r["metadata"]["text"] for r in results if r.get("metadata") and "text" in r["metadata"]]
        
        if not texts:
            return "I don't know based on the provided documents."

        context = "\n---\n".join(texts)

        # Strict prompt for factual answers only
        prompt = f"""
You are the official spokesperson for Muhammad Zubair. 

Rules:
1. Provide a direct, factual, and confident answer based ONLY on the context.
2. NEVER mention "the context," "the matrix," or "the documents."
3. LINK RULE: Only include a link if it is EXPLICITLY provided in the context below. 
4. NO HALLUCINATION: Do not make up link text like "View Power BI Expertise" or "Click here for more" if a URL is not present in the context.
5. If a real URL (starting with https) exists for the topic, format it as: [Link Description](URL).
6. If no URL exists for a specific skill, just describe the skill and stop.
7. Speak in the third person.

Context:
{context}

Question: {query}
Answer:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()
