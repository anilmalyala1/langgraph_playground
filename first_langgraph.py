from duckduckgo_search import DDGS
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.prebuilt import ToolNode
from typing import Sequence, TypedDict, Annotated
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

load_dotenv(find_dotenv(), override=True)

# Your Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Weather in {city}: Sunny, 25°C"

@tool
def search_web(search_str: str) -> str:
    """Search internet and return a compact list of top results"""
    # Keep tool outputs short to avoid token bloat
    print("---Web Search----")
    with DDGS() as ddgs:
        results = list(ddgs.text(search_str, region="sg-en", max_results=5))
    short = [
        f"- {r.get('title','(no title)')} -> {r.get('href','')}"
        for r in results
    ]
    print(short)
    return "Top results:\n" + "\n".join(short)

# Create tool node
tool_node = ToolNode([get_weather,search_web])



# Define your state
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]

# Define your nodes
def agent_node(state: AgentState):
    """Ask the LLM; it may return tool_calls."""
    messages = list(state["messages"])
    # (Optional) steer the LLM to finish only when all subtasks are handled
    sys = SystemMessage(
        content=(
            "You have tools. If the user asks multiple things, call tools step-by-step "
            "until all parts are handled. When done, respond with a final answer."
        )
    )
    if not messages or messages[0].type != "system":
        messages = [sys] + messages

    tools=[get_weather,search_web]
    llm_with_tools = model.bind_tools(tools)
    ai = llm_with_tools.invoke(messages)
    return {"messages": [ai]}


def route_from_agent(state: AgentState) -> str:
    """If the LLM asked for a tool, go to tools; else end."""
    last = state["messages"][-1]
    # LangChain AIMessage has .tool_calls when the model requested tools
    has_calls = getattr(last, "tool_calls", None)
    return "tools" if has_calls else "end"

# Create the graph
workflow = StateGraph(AgentState)



workflow.add_edge(START,"agent")

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# key change: conditional edges from agent -> tools or END
workflow.add_conditional_edges(
    "agent",
    route_from_agent,
    {"tools": "tools", "end": END},
)
workflow.add_edge("tools", "agent")
#workflow.add_edge("tools", END)



# Compile
app = workflow.compile()

# Use it
result = app.invoke({
    "messages": [{"role": "user", "content": "Hello Gemini!, How is the weather in Hyderabad and where is best place to stay in Hyderabad ?"}],
    "next": "agent"
})


print(result)
