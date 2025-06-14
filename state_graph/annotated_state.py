from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List , Annotated
import operator


class CustomState(BaseModel):
    counter : int
    total : Annotated[int , operator.add]
    history : Annotated[List[int], operator.concat]


def increment(State : CustomState) -> CustomState:
    incremented =  State.counter+1
    return {
        "counter": incremented,
        "total": incremented,
        "history":  [incremented]
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