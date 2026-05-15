# python -m src.vectorrag.db
import os

from src.vectorrag.config import MyConfig as mcfg
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def indexing(document,
             collection_name=mcfg.collection_name,
             persist_directory=mcfg.persist_directory,
             embd_model=mcfg.embd_model):
    embed = OllamaEmbeddings(model=embd_model)
    vectorstore = Chroma.from_documents(
        documents=document,
        embedding=embed,
        collection_name=collection_name,
        persist_directory=persist_directory)
    
    # vectorstore.persist()
    
def loading_vectorstore(collection_name=mcfg.collection_name,
                        persist_directory=mcfg.persist_directory,
                        embd_model=mcfg.embd_model):
    embed = OllamaEmbeddings(model=embd_model)
    vectorstore = Chroma(
        embedding_function=embed,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    return vectorstore

def retrieve_query(query, vectorstore, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    content_list = []
    for index, doc in enumerate(docs, start=1):
        content = f"""[Document {index}]
text : {doc.page_content}
            
metadata : {doc.metadata}

"""
        content_list.append(content)

    contents='\n'.join(content_list)
    return contents

    
        
if __name__ == "__main__":
    print("* testing vectorrag.db")
    
    from src.vectorrag.data_processing import data_processing, documents_dict
    from src.vectorrag.config import MyConfig as mcfg
    
    # data processing
    # origin_docs = documents_dict(mcfg.data_path)
    # fin_docs = data_processing(origin_docs, 300, 50)
    # print(fin_docs)
        
    # db indexing
    # indexing(fin_docs,mcfg.collection_name,mcfg.persist_directory)
    # vectorstore = loading_vectorstore()
    # print(vectorstore._collection)
    
    # vector db data 저장 형식 확인
    vectorstore = loading_vectorstore()
    saved_collection = vectorstore._collection
    saved_data = saved_collection.get(ids=None,
                                      limit=1,
                                      include=['embeddings', 'documents', 'metadatas'])
    saved_data["embeddings"] = saved_data["embeddings"][0][:10]
    print("[SAVED_DATA IN VECTORDB]")
    print(saved_data)