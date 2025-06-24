from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from rag_setup_and_chain import retriever
from typing import Annotated, Sequence, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")


reteriver_tool = create_retriever_tool(
    retriever, 
    "reteriver_tool",
    """
    Information related to USICT - univeristy school of information communication and technology on topics such as "
    - Fees
    - Courses offered
    - About insitute/college
    - Placement/Jobs offered
    """

)

@tool 
def off_topic():
    """Catch all the questions that are not related to usict  univeristy school of information communication and technology 
    a college that comes under the university: ipu aka indraprashta univeristy aka Guru gobind singh indraprastha university"""
    return "Forbidden  - Do not repsond to the user"

tools = [reteriver_tool, off_topic]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
    message= state['messages']
    model = ChatGoogleGenerativeAI(temperature=0.9 , model="gemini-2.0-flash").bind_tools(tools)
    response = model.invoke(message)
    return {
        "messages": [response]
    }

def should_continue(state) -> Literal["tools", END]:
    message= state["messages"]
    last_message =  message[-1]
    if last_message.tool_calls:
        return "tools"
    return END

tool_node =ToolNode(tools)

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent" , should_continue)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")
app = graph.compile()


while True:
    msg = input(">")
    res = app.invoke({
    "messages": [HumanMessage(msg)]
    })
    print(res['messages'][-1].content)