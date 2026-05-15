# python -m src.vectorrag.query_answer
import os
import re
import json

from src.vectorrag.config import MyConfig as mcfg
from src.vectorrag.prompts import VectorRAG_Prompt, make_prompt
from src.vectorrag.db import loading_vectorstore, retrieve_query
from langchain_ollama import ChatOllama


def query_answering(query,
                    vectorstore,
                    llm_model=mcfg.ollama_model,
                    k=5):
    
    llm = ChatOllama(model=llm_model)
    prompts=VectorRAG_Prompt()
    contents = retrieve_query(query, vectorstore, k=k)

    query_answering_prompt = make_prompt(sys_prompt=prompts.RetrievalQA_sys_prompt,
                                  human_prompt=prompts.RetrievalQA_human_prompt,
                                  context=contents)
    
    qna_chain = query_answering_prompt | llm
    response = qna_chain.invoke({"input": query})

    return response
    
        
if __name__ == "__main__":
    print("* testing query_answer")
    vectorstore = loading_vectorstore()
    query = "온디바이스 AI가 부상하게 된 이유는?"
    
    # vector db data 검색 확인
    print("[RETRIEVED DOCUMENTS]")
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    print(docs_with_scores)
    
    print("[RESPONSE]")
    response = query_answering(query, vectorstore, k=3)
    print(response.content)