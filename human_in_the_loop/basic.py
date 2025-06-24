#generate a tweet take feedbacks 
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage , HumanMessage
from langgraph.graph import add_messages, END, StateGraph
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def generate(state : BasicChatState):
    return {
        "messages": state["messages"] + [llm.invoke(state["messages"])]
    }

def get_review(state: BasicChatState):
    print(state["messages"][-1].content)
    review = input("what review do you want to give?")
    return {
        "messages": state["messages"] + [HumanMessage(content=review)]
    }

def review(state: BasicChatState):
    content = state["messages"][-1].content
    print(content)
    d = input("\nDo you wnat to keep this?")
    if(d.lower() == "y"):
        return "post"
    else:
        return "get_review"
    

def post(state: BasicChatState):
    print("posting........")
    print(state["messages"][-1].content)
    print("Success")
    
graph = StateGraph(BasicChatState)

graph.add_node("generate", generate )
graph.add_node("review", review)
graph.add_node("get_review", get_review)
graph.add_node("post", post)
graph.set_entry_point("generate")
graph.add_conditional_edges("generate", review)
graph.add_edge("get_review", "generate")
graph.add_edge("post", END)
app = graph.compile()

response = app.invoke({
    "messages": [HumanMessage(content="Write a twitter post about how svelte it the best frontend framework")]
})