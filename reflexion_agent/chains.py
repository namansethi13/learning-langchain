from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder
import datetime
from schema import AnswerQuestion, RevisedAnswer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAK")
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

#actor prompt templae this will be shared by both responder and  reviosr
actor_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system" ,
        """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
    ), 
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Answer the user's question above using the required format.",
    )
    
]).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed 250 word answer"
)

llm = ChatGoogleGenerativeAI(temperature=0.8, model="gemini-2.5-flash-preview-04-17") 

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
) 
# | {
#     "answer" : lambda x : x.answer, 
#     "search_queries" : lambda x : x.search_queries,
#     "reflection" : lambda x : {
#         "missing" : x.reflection.missing,
#         "superfluous": x.reflection.superfluous
#     }
# }

revise_prompt= """
Revise your new answer using the new information.
    - You should use previous critique to add important information to your answer.
    - You must include numerical citations in your revised response so that the information can be verified.
    - Include a references section, this section doesn't count in the word limit it should be in the form 
        -- [1] www.example.com accessed 25 may 2025
        -- [2] www.example2.com accessed 26 may 2025
        -- [3] www.example3.com accessed 27 may 2025
    - Use the previous critique to remove superfluous in your answer, make sure you answer (excluding the references part) does not exceed 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_prompt
) | llm.bind_tools(tools=[RevisedAnswer] , tool_choice="RevisedAnswer")

response = first_responder_chain.invoke(
    {
        "messages" : [HumanMessage(content="Write me a blog post on how sveltkit is the best frontend framework")]
    }
)

print(response.additional_kwargs)
