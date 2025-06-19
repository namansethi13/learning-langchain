from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph import add_messages,StateGraph, END
from langchain_core.messages import AIMessage , HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
sqlite_conn= sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

tavily = TavilySearchResults()
memory = SqliteSaver(sqlite_conn)

tools=[tavily]
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools )
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state : BasicChatState):
    return{
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


def tool_router(state : BasicChatState):
    last_message = state["messages"][-1]
    if(hasattr(last_message, "tool_calls")) and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot" , tool_router)
graph.add_edge("tool_node" , "chatbot")
app = graph.compile(checkpointer=memory)
config = {
    "configurable":{
        "thread_id": 1
    }
}

while True:
    userInput = input(">")
    res = app.invoke({
        "messages": HumanMessage(content=userInput)
    }, config=config)
    print(res['messages'][-1].content)