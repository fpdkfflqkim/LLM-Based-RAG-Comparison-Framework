# python -m src.vectorrag.prompts

import os
from src.vectorrag.config import MyConfig
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

class VectorRAG_Prompt:
    def __init__(self, folder_path=MyConfig.prompt_path):
        self.folder_path = folder_path
        
        self.RetrievalQA_sys_prompt = self.read_prompt(self.folder_path, "RetrievalQA_sys_prompt.txt")
        self.RetrievalQA_human_prompt = self.read_prompt(self.folder_path, "RetrievalQA_human_prompt.txt")
        
        
    def read_prompt(self,
                    folder_path,
                    prompt_name):
        
        prompt_path = os.path.join(folder_path, prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        
        return prompt
    
def make_prompt(sys_prompt,
                human_prompt,
                examples=None,
                format_instructions=None,
                context=None):
    # sys_prompt
    SYS_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
    
    # human_prompt
    
    partial_variables = {}
    if examples:
        partial_variables["examples"] = examples
    if format_instructions:
        partial_variables["format_instructions"] = format_instructions
    if context:
        partial_variables["context"] = context
        
    HUMAN_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["input"],
            partial_variables=partial_variables,
            template=human_prompt
            )
        )
    fin_prompt = ChatPromptTemplate.from_messages([SYS_prompt, HUMAN_prompt])
    
    return fin_prompt
        
if __name__ == "__main__":
    print("* testing prompts")
    prompts=VectorRAG_Prompt()
    print(prompts.index_human_prompt)