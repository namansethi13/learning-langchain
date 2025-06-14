from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List


class CustomState(BaseModel):
    counter : int
    total : int
    history : List[int]


def increment(State : CustomState) -> CustomState:
    return {
        "counter": State.counter+1,
        "total": State.total+ State.counter+1,
        "history":  State.history + [State.counter]
    }

def should_continue(State : CustomState)-> str:
    if State.counter > 5:
        return 'done'
    return 'go'

graph = StateGraph(CustomState)
graph.add_node("increment" , increment)
graph.add_conditional_edges("increment", 
should_continue,
{
    "go" : "increment",
    "done": END
}
)

init = {
    "counter":0,
    "total":0,
    "history": []
}
graph.set_entry_point("increment")
app = graph.compile()
result = app.invoke(init)
print(result)