from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Import your data loader to ensure example usage works
try:
    from src.data_loader import load_all_documents
except ImportError:
    # This handles cases where the script is run from different folders
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 250):
        """
        Handles splitting text into manageable chunks and converting them into vector embeddings.
        Chunk overlap is set to 250 to ensure end-of-file data (like hobbies) isn't lost.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Splits documents into smaller chunks. 
        The separators are prioritized to keep bullet points and sentences together.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Priority: Paragraphs -> New Lines -> Bullets -> Sentences
            separators=["\n\n", "\n", "•", "- ", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Converts text chunks into numerical vectors.
        """
        if not chunks:
            print("[WARNING] No chunks provided for embedding.")
            return np.array([])
            
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        
        # normalize_embeddings improves the accuracy of the FAISS search
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        print(f"[INFO] Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings

# Example usage for testing
if __name__ == "__main__":
    # 1. Load documents from the data folder
    docs = load_all_documents("data")
    
    if docs:
        # 2. Initialize pipeline
        emb_pipe = EmbeddingPipeline()
        
        # 3. Process
        chunks = emb_pipe.chunk_documents(docs)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        print(f"\n[SUCCESS] Successfully processed {len(docs)} files.")
        print(f"[DEBUG] First chunk preview: {chunks[0].page_content[:100]}...")
    else:
        print("[ERROR] No documents found in 'data' folder. Please check your files.")