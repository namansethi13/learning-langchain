from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate , MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
llm = ChatGoogleGenerativeAI(temperature=0.9, model="gemini-2.0-flash")
system_prompt = "You are a helpful ai bot for children called joe the ai"

prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}")
])

pipe = (
    {"query" : lambda x : x['query'],
    "history": lambda x : x["history"]
    }
    |
    prompt_template
    |
    llm
)

chat_map = {} #stores the chat history key wise represents a keyed data base
def get_chat_history(session_id : str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map.keys():
        chat_map[session_id] = InMemoryChatMessageHistory()
        chat_map[session_id].add_user_message("Hey!")
        chat_map[session_id].add_ai_message("Hi my name is joe the ai, how can I help you")
        chat_map[session_id].add_user_message("I am in 2nd grade and my name is naman")
        chat_map[session_id].add_ai_message("Great! What would you like to lean today? Do you pehaps need help with your home work ")

    return chat_map[session_id]

pipeline_with_history = RunnableWithMessageHistory(
    pipe , 
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)


while True:
    query = input(">")
    res = pipeline_with_history.invoke(
        {"query": query},
        config={"session_id" : "123"}
    )
    print("ai:",res.content)