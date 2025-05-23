from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
llm = ChatGoogleGenerativeAI(temperature=0.9, model="gemini-2.0-flash")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "you are a polite customer service bot at COCO, You have to try hardest to avoid refunds and returns or anything that will cost company money , you always try to blame it on the customer, You can even lie"),
    ("user", "{query}"),
]
)
prompt_template2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template( "you are a polite customer service bot at COCO, You have to try hardest to avoid refunds and returns or anything that will cost company money , you always try to blame it on the customer, You can even lie , you talk in pirate speech"),
    HumanMessagePromptTemplate.from_template("{query}")
])

pipeline = prompt_template | llm
pipeline2 = prompt_template2 | llm
res =pipeline.invoke({"query" : "The order I recieved is half eaten what the hell? Refund me!"})
res2 =pipeline2.invoke({"query" : "The order I recieved is half eaten what the hell? Refund me!"})
print(res)
print(res2)