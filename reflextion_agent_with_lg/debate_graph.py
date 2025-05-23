from typing import List , Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from religion_chains import atheist_chain, beliver_chain
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

ITERATIONS = 3
BELIVER  = "beliver"
ATHEIST  = "atheist"

def atheist_node(state):
    return atheist_chain.invoke(
        {
            "messages": state
        }
    )

def beliver_node(state):
    return beliver_chain.invoke(
        {
            "messages": state
        }
    )

def should_continue(state):
    if len(state) > ITERATIONS*2:
        return END
    return BELIVER


graph = MessageGraph()
graph.add_node(ATHEIST, atheist_node)
graph.add_node(BELIVER, beliver_node)
graph.set_entry_point(BELIVER)

graph.add_conditional_edges(ATHEIST, should_continue )
graph.add_edge(BELIVER, ATHEIST )

app = graph.compile()

res =app.invoke(HumanMessage(content="Start the debate"))
print(res)