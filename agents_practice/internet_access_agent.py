from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder,SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
import pytz
from datetime import datetime
from langchain.tools import tool
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
os.environ["TAVILY_API_KEY"] = os.getenv("TAK")

llm = GoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash") 
search_tool = TavilySearchResults()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are an india focuced news summerizer agent that gives summary of top 5 news according to the category user tells you for today"""),
    HumanMessagePromptTemplate.from_template("{input}")
])

@tool
def get_today_date(format : str = "%Y-%m-%d"):
    """
    Outputs todays date 
    """
    return datetime.now().strftime(format)


tools= [search_tool, get_today_date]
agent = initialize_agent(tools=tools , llm=llm , agent_type="zero-shot-react-description", verbose=True)

chain = prompt | agent

res = chain.invoke({"input" : "geo politics"})

print(res)