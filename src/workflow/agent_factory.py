from src.agents.agent_base import AgentBase
from src.agents.math import Math
from src.workflow.llm import LLM

class AgentFactory(object):
    def __init__(self):
        self.agents: dict[str, AgentBase] = {}

    def create_agent(self, name: str, system_prompt: str, llm: LLM, user_prompt: str=None, tools=None, structured_output=None):
        self.agents[name] = Math(system_prompt, llm, tools, structured_output)