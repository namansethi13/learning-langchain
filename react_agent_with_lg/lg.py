from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from datetime import datetime
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
os.environ["TAVILY_API_KEY"] = os.getenv("TAK")
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

llm = ChatGoogleGenerativeAI(temperature=0.8, model="gemini-2.5-flash-preview-04-17") 



search_tool = TavilySearchResults()

@tool
def get_today_date(format : str = "%Y-%m-%d"):
    """
    Outputs todays date 
    """
    return datetime.now().strftime(format)

tools = [search_tool, get_today_date]
prompt = """
You are an india focuced news summerizer agent that gives summary of top 5 news according to the category user tells you for today
You can access tools like:
- 'search_tool' for getting recent or live web data.
- 'get_today_date' to find the current date.
Use these tools wisely to help the user.
"""
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt
)
inputs = {"messages": [{"role": "user", "content": "Society, Ethics & Culture "}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)