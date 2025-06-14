from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")


generator_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a very popular twitter influencer, your task is to generate tech tweets, if user provides critique then revise the pervious versions"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a social media pro, who knows how to make tweets go viral, given a tweet generate critique and recomendation to make the tweet go viral"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = GoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash") 

generation_chain = generator_prompt | llm 
reflection_chain = reflector_prompt | llm 
