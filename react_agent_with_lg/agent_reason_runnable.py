from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool , create_react_agent
from datetime import datetime
from langchain_community.tools import TavilySearchResults
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")
os.environ["TAVILY_API_KEY"] = os.getenv("TAK")

llm = GoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash") 
search_tool = TavilySearchResults()

@tool
def get_today_date(format : str = "%Y-%m-%d"):
    """
    Outputs todays date 
    """
    return datetime.now().strftime(format)

tools = [search_tool, get_today_date]

create_react_agent(tools=tools, llm=llm , prompt="")
