import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store", embedding_model=None, chunk_size=1500, chunk_overlap=300):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

    def build_from_documents(self, documents: List[Any]):
        emb_pipe = EmbeddingPipeline(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        emb_pipe.model = self.embedding_model
        
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim) # Reliable for small datasets
        self.index.add(embeddings.astype("float32"))

        self.metadata = [{"text": chunk.page_content} for chunk in chunks]
        self.save()

    def save(self):
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> bool:
        if not os.path.exists(self.faiss_path) or not os.path.exists(self.meta_path):
            return False
        self.index = faiss.read_index(self.faiss_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        if self.index is None: return []
        D, I = self.index.search(query_embedding.astype("float32"), top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({"metadata": self.metadata[idx], "distance": float(dist)})
        return results

    def update_from_new_documents(self, data_dir: str = "data"):
        from src.data_loader import load_all_documents
        all_docs = load_all_documents(data_dir)
        existing_texts = set(m['text'] for m in self.metadata)
        new_docs = [doc for doc in all_docs if doc.page_content not in existing_texts]

        if new_docs:
            emb_pipe = EmbeddingPipeline(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            emb_pipe.model = self.embedding_model
            chunks = emb_pipe.chunk_documents(new_docs)
            embeddings = emb_pipe.embed_chunks(chunks)
            if self.index is None: self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype("float32"))
            self.metadata.extend([{"text": c.page_content} for c in chunks])
            self.save()