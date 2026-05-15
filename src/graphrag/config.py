# python -m graphrag.config

class MyConfig:
    data_path = "./src/graphrag/input_doc"
    prompt_path = "./src/graphrag/prompt"
    ollama_model = "gemma4:31b-cloud" # "gemma3:1b"
    embd_model = "embeddinggemma"
    
    
if __name__ == "__main__":
    print("* testing graphrag.config")
    print(MyConfig.prompt_path)
    print(MyConfig.ollama_model)
    print(MyConfig.embd_model)