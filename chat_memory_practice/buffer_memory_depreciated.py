from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
import os
memory = ConversationBufferMemory(return_messages=True) # making it true forces it to return it in message objects and not in string. Chat LLMs expects message objects
os.environ["GOOGLE_API_KEY"] = os.getenv("GAK")

memory.save_context(
    {"input": "Hey!"},
    {"output": "Hi my name is joe the ai, how can I help you"}
)
memory.save_context(
    {"input": "I am in 2nd grade"},
    {"output": "Great! what would you like to lean today? Do you pehaps need help with your home work "}
)
memory.save_context(
    {"input": "Well yeah! I want to lean about our solar system"},
    {"output": """
    🌞 The Solar System is like a big family in space. At the center is the Sun — a huge, hot ball of fire that gives us light and heat.

🌍 Around the Sun, there are planets that go around it in circles. It’s kind of like how you might go around a merry-go-round!

Here are the 8 planets, from the one closest to the Sun to the farthest:

Mercury - the smallest and closest to the Sun.

Venus - super hot and bright!

Earth - that's where we live!

Mars - the red planet.

Jupiter - the biggest planet!

Saturn - it has beautiful rings!

Uranus - it spins on its side.

Neptune - the farthest and very cold.

🌟 There are also moons, which go around the planets (Earth has one moon!). Plus, there are asteroids (space rocks) and comets (icy space balls with tails).

So the solar system is like a big space neighborhood, with the Sun in the middle and all the planets, moons, and space stuff going around it.
    """}
)

memory.load_memory_variables({})
llm = ChatGoogleGenerativeAI(temperature=0.9, model="gemini-2.0-flash")
chain = ConversationChain(
    llm = llm,
    memory = memory,
    verbose= False
)

if __name__ == "__main__":
    while True:
        prompt = input(">")
        memory.chat_memory.add_user_message(prompt)
        res = chain.invoke({"input" : prompt})
        memory.chat_memory.add_ai_message(res['response'])
        print(res['response'])
