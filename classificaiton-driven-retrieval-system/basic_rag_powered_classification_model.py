from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage , HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langgraph.graph import add_messages , StateGraph, END
from pydantic import BaseModel,Field
from rag_setup_and_chain import llm , retriever, rag_chain
import random
class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: bool

class GradQuestion(BaseModel):
    """
    Boolean value to check whether the question asked is related to USICT college
    """
    score : bool = Field(
        description="Return True if the question is about USICT college (e.g. general info about the usict college, fees, courses, placements, etc.), else False.",
        example=True
    )

def question_classifier(state: AgentState):
    question = state['messages'][-1].content
    system = """
    You are a classifier agent and your job is to classify the question as on topic i.e. about the
    college if it mathes the following topic:
    - Fees
    - Courses offered
    - About insitute/college
    - Placement/Jobs offered
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system" , system),
            ("human", "User question {question}")
        ]
    )
    structured_llm = llm.with_structured_output(GradQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({
        "question" :question
    })
    state["on_topic"] = result.score
    return state

def on_topic_router(state : AgentState):
    on_topic= state['on_topic']
    if on_topic:
        return "retrieve"
    return "off_topic"

def retrieve(state: AgentState):
    question = state['messages'][-1].content
    document = retriever.invoke(question)
    state["documents"] = document
    return state

def generate_answer(state: AgentState):
    question = state['messages'][-1].content
    document = state["documents"]
    generation = rag_chain.invoke({
        "content":document,
        "question":question
    })
    state['messages'].append(generation)

def off_topic(state: AgentState):
    messages = [
    "I'm afraid I don't have the answer to that.",
    "Sorry, I can't help with that one.",
    "That's outside my current knowledge.",
    "I'm not sure how to respond to that.",
    "Unfortunately, I don't know the answer.",
    "I don't have enough information to answer that.",
    "Hmm, that's not something I can answer right now.",
    "Apologies, I can't provide an answer to that.",
    "That's a bit beyond my capabilities.",
    "I wish I could help, but I don't have that info.",
    "Sorry, that's not something I can explain at the moment.",
    "I'm not able to assist with that question.",
    "That query is a bit out of my depth.",
    "I can't give you a reliable answer to that.",
    "I'm still learning and can't answer that yet."
    ]
    msg = random.choice(messages)
    state['messages'].append(AIMessage(content=msg))
    return state

graph = StateGraph(AgentState)

graph.add_node("question_classifier" , question_classifier)
graph.add_node("off_topic" , off_topic)
graph.add_node("retrieve", retrieve)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("question_classifier")

graph.add_conditional_edges("question_classifier", on_topic_router)
graph.add_edge("retrieve", "generate_answer")
graph.add_edge("generate_answer" ,END)
graph.add_edge("off_topic", END)

app = graph.compile()

while True:
    question = input(">")
    question = HumanMessage(question)
    res = app.invoke(input={
        "messages": [question]
    })

    print("AI: ", res['messages'][-1].content)