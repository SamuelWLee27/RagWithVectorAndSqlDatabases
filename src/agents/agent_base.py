from abc import ABC, abstractmethod
from typing import Literal

from langgraph.graph import MessagesState, END, StateGraph, START
from langchain.messages import ToolMessage
from src.workflow.llm import LLM

class AgentBase(ABC):
    def __init__(self, system_prompt: str, llm: LLM, tools, user_prompt: str=None, structured_output=None):
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        llm_wrapper = llm
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm = llm_wrapper.get_llm_with_tools(tools)
        if structured_output is not None:
            self.llm = llm_wrapper.get_llm_with_structured_output(structured_output, llm=self.llm)

        self.subgraph = self.build_subgraph()

    @abstractmethod
    def call_llm(self, state: MessagesState):
        pass

    def tool_node(self, state: MessagesState):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    @staticmethod
    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    def build_subgraph(self):
        subgraph_builder = StateGraph(MessagesState)
        subgraph_builder.add_node("call_llm", self.call_llm)
        subgraph_builder.add_node("tool_node", self.tool_node)
        subgraph_builder.add_edge(START, "call_llm")
        subgraph_builder.add_conditional_edges(
            "call_llm",
            self.should_continue,
            ["tool_node", END]
        )
        subgraph_builder.add_edge("tool_node", "call_llm")
        return subgraph_builder.compile()

    @abstractmethod
    def call_subgraph(self, state: MessagesState):
        pass
