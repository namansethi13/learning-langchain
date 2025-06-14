from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")


atheist_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are an atheist, debating against the notion there is a place for god in 21st century respond within 120 words, and counter arguments and create new arguments"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

beliver_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a beliver, debating for the notion there is a place for god in 21st century respond within 120 words, and counter arguments and create new arguments"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = GoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash") 

atheist_chain = atheist_prompt | llm 
beliver_chain = beliver_prompt | llm 
