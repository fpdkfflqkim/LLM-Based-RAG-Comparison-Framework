# python -m test.comparing_methods
if __name__ == "__main__":
    print("* testing all methods")
    from src.vectorrag.config import MyConfig as mcfg
    
    query_list = [
        "델타테크는 어떤 회사인가?",
        "업계에서 부정적인 평가를 받고 있는 회사와 그 이유는?",
        "델타테크 창업자가 AlphaAI 퇴사 후 설립한 회사에 투자한 기관은?"
        ]
    
    # import for plain LLM
    from langchain_ollama import OllamaLLM

    # import for vector rag
    from src.vectorrag.db import loading_vectorstore
    from src.vectorrag.query_answer import query_answering as vectorrag_query_answering
    
    # import for graph rag
    from src.graphrag.query_answer import query_answering as graphrag_query_answering
    from src.graphrag.retrieve import graph_retriever, vector_retriever
    
    # import for hybrid search (vector + graph)
    from src.graphrag.query_answer import query_answering_hybrid as graphrag_query_answering_hybrid
    
    # Vector DB connect
    vectorstore = loading_vectorstore()
    
    for query in query_list:
        
        # plain LLM
        llm_query = query + "\n 한 문단 정도로 간단하게 답변하라."
        no_rag_llm = OllamaLLM(
            model=mcfg.ollama_model,
            )
        no_rag_respose = no_rag_llm.invoke(llm_query)
    
        
        # Vector RAG
        vectorrag_response = vectorrag_query_answering(query, vectorstore, k=3)
        
        # Graph RAG
        graphrag_response = graphrag_query_answering(query)
        
        # Hybrid RAG
        graphrag_response_hybrid = graphrag_query_answering_hybrid(query, k=3)

        # response 비교
        print("="*60)
        print("[Plain LLM Response]")
        print(no_rag_respose)
        
        print()
        
        print("[Vector RAG Response]")
        print(vectorrag_response.content)
        # print(vectorstore.similarity_search_with_score(query, k=3)) # related documents
        
        print()
        
        print("[Graph RAG Response]")
        print(graphrag_response.content)
        # print(graph_retriever(query)) # related subgraph
        
        print()
        
        print("[Hybrid RAG Response]")
        print(graphrag_response_hybrid.content)
        # print(graph_retriever(query))) # related subgraph
        # print(vector_retriever(query)) # related documents
        print("="*60)