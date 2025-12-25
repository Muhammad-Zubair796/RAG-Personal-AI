# app.py

from src.search import RAGSearch

if __name__ == "__main__":
    # Initialize RAGSearch (loads index, builds if missing, automatically adds new documents)
    rag_search = RAGSearch()

    # Example queries – you can replace or loop through multiple questions
    queries = [
        "Does he know Power BI?",
        "Can he do SEO?",
        "What games has he developed?"
    ]

    for query in queries:
        print(f"Question: {query}")
        answer = rag_search.search_and_summarize(query, top_k=10)
        print("Answer:", answer)
        print("-" * 80)
