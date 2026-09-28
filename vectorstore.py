import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_index(self, chunks, progress_callback=None, batch_size=64):
        self.chunks = chunks
        self.index = None
        total = len(chunks)

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")

            if self.index is None:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        print("Index built with", total, "chunks.")

    def search(self, query, top_k=10):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            if i != -1:
                results.append(self.chunks[i])
        return results