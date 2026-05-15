# python -m src.vectorrag.config
class MyConfig:
    data_path = "./src/vectorrag/input_doc"
    ollama_model = "gemma4:31b-cloud" # "gemma3:1b"
    embd_model = "embeddinggemma"
    prompt_path = "./src/vectorrag/prompt"
    persist_directory = "./src/vectorrag/db"
    collection_name = "vstore_vs_graph_py"
    
    
if __name__ == "__main__":
    print("* testing config")
    print(MyConfig.data_path)
    print(MyConfig.ollama_model)
    print(MyConfig.embd_model)