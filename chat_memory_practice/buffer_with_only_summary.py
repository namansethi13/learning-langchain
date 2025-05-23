from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate , MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory # this is the base class that can be used to implement buffer window as in newer langchain version there is no implmentation for it
from langchain_core.messages import BaseMessage
from pydantic import BaseModel , Field
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
llm = ChatGoogleGenerativeAI(temperature=0.9, model="gemini-2.0-flash")


class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages : list[BaseMessage] = Field(default_factory=list)
    k : int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
    def add_messages(self, messages : list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:];
    def clear() -> None:
        self.messages = []


system_prompt = "You are a helpful ai bot for a person called naman with description by another ai :  'You are a technically inclined individual with a strong interest in web development, particularly using modern frameworks like SvelteKit, TypeScript, Tailwind CSS, and DaisyUI. You're actively enhancing your skills through projects, such as building a blog website, and learning key concepts in computer science like data structures and algorithms—evident from your LeetCode practice and contest participation.'"

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
def get_chat_history(session_id : str) -> BufferWindowMessageHistory:
    if session_id not in chat_map.keys():
        chat_map[session_id] = BufferWindowMessageHistory(k=8)
        chat_map[session_id].add_user_message("Hey!")
        chat_map[session_id].add_ai_message("Hi my name is joe the ai, how can I help you")
        chat_map[session_id].add_user_message("my name is naman")
        chat_map[session_id].add_ai_message("Great! What would you like to learn today?")

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

