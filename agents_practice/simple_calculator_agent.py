from langchain.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder,SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
import os
from langchain_core.runnables.base import RunnableSerializable
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

from dotenv import load_dotenv

load_dotenv()
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are an agent and you should use tools provided to answer the user.
You should write the output of the tool in the scratch-pad provided below and keep doing so until scratch-pad has the answer.
If the scratchpad has the answer, you should return it without a tool call."""),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="scratchpad"),
    HumanMessagePromptTemplate.from_template("{input}")
])
finished = False
@tool
def add(x: float, y: float) -> float:
    """Takes two float numbers x and y and returns x + y"""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Takes two float numbers x and y and returns x * y"""
    return x * y

@tool
def subtract(x: float, y: float) -> float:
    """Takes two float numbers x and y and returns x - y"""
    return x - y

@tool
def divide(x: float, y: float) -> float:
    """Takes two float numbers x and y and returns x / y if y != 0 else returns 0"""
    return 0 if y == 0 else x / y

@tool
def exponent(x: float, y: float) -> float:
    """Takes two float numbers x and y and returns x raised to the power y"""
    return x ** y

@tool
def give_output(output : str)-> str:
    """Gives the user exact string output that is passed in the function as an argument"""
    print(output)
    finished = True

tools = [add, multiply, subtract, divide, exponent, give_output]

# Instantiate the LLM
llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash") 


query = "what is two times 3 added with 5 to the 4th power"

agent : RunnableSerializable = (
    {
        "input": lambda x : x["input"],
        "history": lambda x : x["history"],
        "scratchpad": lambda x : x.get("scratchpad", [])
    }
    |
    prompt
    |
    llm.bind_tools(tools , tool_choice="any")
)

times = 5


name_to_tool_mapping = {
    tool.name: tool.func for tool in tools
}
tool_call = agent.invoke({
    "input":query,
    "history":[],
})
while times > 0 and not finished:
    print(tool_call)
    tool_exec = name_to_tool_mapping[tool_call.tool_calls[0]['name']](**tool_call.tool_calls[0]['args'])

    tool_says = ToolMessage(
        content=f"The output of the tool call is: {tool_exec}",
        tool_call_id=f"{tool_call.tool_calls[0]['id']}"
    )
    tool_call = agent.invoke({
        "input": query,
        "history": [],
        "scratchpad": [*scratchpad , tool_call , tool_says]
    }
    )

    times = times -1