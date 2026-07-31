from langchain_ollama import ChatOllama
import os

class LLM(object):
    def __init__(self):
        llm_name = os.getenv("LLM_NAME", "llama3.1:8b")
        self.llm = ChatOllama(model=llm_name, temperature=0)

    def get_base_llm(self):
        return self.llm

    def get_llm_with_tools(self, tools, llm=None):
        return llm.bind_tools(tools=tools) if llm else self.llm.bind_tools(tools=tools)

    def get_llm_with_structured_output(self, structured_output, llm=None):
        return llm.with_structured_output(structured_output) if llm else self.llm.with_structured_output(structured_output)