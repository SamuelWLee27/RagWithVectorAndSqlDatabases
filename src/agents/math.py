

from langgraph.graph import MessagesState
from langchain.messages import SystemMessage

from src.workflow.llm import LLM
from src.agents.agent_base import AgentBase

class Math(AgentBase):
    def __init__(self, system_prompt: str, llm: LLM,tools=None, structured_output=None):
        AgentBase.__init__(self, system_prompt=system_prompt, llm=llm, tools=tools, structured_output=structured_output)

    def call_llm(self, state: MessagesState):
        return {
            "messages": [
                self.llm.invoke(
                    [
                        SystemMessage(
                            content=self._system_prompt
                        )
                    ]
                    + state["messages"]
                )
            ]
        }

    def call_subgraph(self, state: MessagesState):
        subgraph_output = self.subgraph.invoke({"messages": state['messages']})
        return {"messages": subgraph_output["messages"]}