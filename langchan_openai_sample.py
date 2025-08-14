# pip install langchain-openai duckduckgo-search python-dotenv
import os, json
from dotenv import load_dotenv, find_dotenv
from typing import Annotated, List
from duckduckgo_search import DDGS

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv(find_dotenv())

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.0-flash",
    temperature=0,
)

# --- Tools ---
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

tools = [get_weather, search_web]
llm_tools = llm.bind_tools(tools)
"""
# --- Run once with automatic tool execution ---
messages: List = [HumanMessage(content="List 5 iconic Singapore hawker dishes. "
                                       )]
                                       
# --- Run once with automatic tool execution ---
messages: List = [HumanMessage(content="What is the weather in hyderabad. "
                                       "Use tools only if you truly need them.")]
                                       """
# --- Run once with automatic tool execution ---
messages: List = [HumanMessage(content="Who is Mahatma Gandhi ?. "
                                       "Use tools only if you truly need them.")]

# 1) Ask the model
ai: AIMessage = llm_tools.invoke(messages)

# 2) If the model asked to call tools, execute them and send results back
if ai.tool_calls:
    tool_msgs = []
    for call in ai.tool_calls:
        name = call["name"]
        args = call.get("args", {}) or {}
        # Find and run the matching tool
        fn = {t.name: t for t in tools}[name]
        result = fn.invoke(args)
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # 3) Give the tool results back to the model for its final answer
    messages = messages + [ai] + tool_msgs
    final_ai: AIMessage = llm.invoke(messages)
    print(final_ai.content)
else:
    # Model answered directly without tools
    print(ai.content)



