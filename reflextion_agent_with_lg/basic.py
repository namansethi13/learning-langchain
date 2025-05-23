from typing import List , Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

ITERATIONS = 3

REFLECT  = "reflect"
GENERATE  = "generate"

def generate_node(state):
    return generation_chain.invoke(
        {
            "messages": state
        }
    )

def reflect_node(state):
    return reflection_chain.invoke(
        {
            "messages": state
        }
    )

def should_continue(state):
    if len(state) > ITERATIONS*2:
        return END
    return REFLECT


graph = MessageGraph()
graph.add_node(REFLECT, reflect_node)
graph.add_node(GENERATE, generate_node)
graph.set_entry_point(GENERATE)

graph.add_conditional_edges(GENERATE, should_continue )
graph.add_edge(REFLECT, GENERATE )

app = graph.compile()

res =app.invoke(HumanMessage(content="Dead internet theory after the ai boom"))
print(res)