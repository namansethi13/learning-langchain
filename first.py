#Objective : to make a very simple chat agent for teaching children ranging 5-10 years old
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
chat_llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.0-flash")

# system_message = SystemMessage(
#     content="You are a helpful assistant for children aged 5-10. You are friendly, patient, and always ready to help them learn new things. You can answer questions, explain concepts, and provide fun facts. Your goal is to make learning enjoyable and engaging"
# )
system_message = SystemMessage(
    content="You are a helpful assistant for university students. You are friendly, patient, and always ready to help them learn new things. You can answer questions, explain concepts, and provide fun facts. Your goal is to make learning enjoyable and engaging"
)

human_message = HumanMessage(
    content="Can you tell me about the solar system?"
)

response = chat_llm.invoke([system_message, human_message])
print(response)