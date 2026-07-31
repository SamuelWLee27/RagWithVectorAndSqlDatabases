from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.messages import  HumanMessage
from src.workflow.agent_factory import AgentFactory
from src.workflow.llm import LLM
from src.tools.fake_tool import multiply

def create_workflow():
    agent_factory = AgentFactory()
    llm = LLM()
    agent_factory.create_agent("nav_agent","You are a helpful assistant tasked with performing arithmetic on a set of inputs.", llm, tools=[multiply])
    builder = StateGraph(MessagesState)
    builder.add_node("node_1", agent_factory.agents["nav_agent"].call_subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)
    graph = builder.compile()
    messages = graph.invoke({"messages": [HumanMessage(content="What is 10 times 10")]})
    for m in messages["messages"]:
        m.pretty_print()

create_workflow()
