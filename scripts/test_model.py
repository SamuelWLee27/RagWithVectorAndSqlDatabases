from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langfuse.langchain import CallbackHandler


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- Ollama: the local model ---
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# --- Tools ---
@tool
def get_stock_level(product: str) -> str:
    """Look up warehouse stock for a Syos product."""
    fake_db = {"drone-x1": 42, "battery-pack": 7}
    return f"{product}: {fake_db.get(product.lower(), 'not found')} units"

tools = [get_stock_level]
tools_by_name = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)

def llm_call(state):
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=("You are a Syos Aerospace assistant who can allow answer questions related to Syos Aerospace."
                                " Don't make things up.")
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["environment", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "Action"

    return END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)

agent_builder.add_edge("environment", "llm_call")

agent = agent_builder.compile()


from IPython.display import Image, display

# with open("graph.png", "wb") as f:
#  f.write(agent.get_graph().draw_mermaid_png())

langfuse_handler = CallbackHandler()

messages = [HumanMessage(content="How many battery-packs do we have?")]
messages = agent.invoke({"messages": messages}, config={"callbacks": [langfuse_handler]})
for m in messages["messages"]:
    m.pretty_print()