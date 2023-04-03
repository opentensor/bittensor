import pinecone
from core.utils import get_ada_embedding

def context_agent(query: str, index: str, n: int):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n,
    include_metadata=True)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)    
    return [(str(item.metadata['task'])) for item in sorted_results]