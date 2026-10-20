"""Abstract vector store interface and Chroma implementation.

Design decision: We hide Chroma behind an ABC (VectorStore) so that swapping to
Weaviate, Qdrant, or Pinecone only requires adding a new subclass and updating
the factory in dependencies.py. The rest of the codebase never imports
chromadb directly.
"""
