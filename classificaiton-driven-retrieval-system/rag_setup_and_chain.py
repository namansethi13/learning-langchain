from langchain.schema import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
embedding = CohereEmbeddings(model="embed-english-light-v3.0")

docs = [
    Document(
        page_content="University School of Information, Communication & Technology (USICT), established in 1999, is one of the premier schools of Guru Gobind Singh Indraprastha University (GGSIPU), Delhi. It was founded with the aim of promoting quality education and research in the fields of engineering and technology. USICT has grown rapidly, offering B.Tech, M.Tech, and Ph.D. programs in IT, CSE, ECE, and Robotics.",
        metadata={"sources": "history.txt"}
    ),
    Document(
        page_content="As of 2024, the annual tuition fee for B.Tech programs at USICT is approximately ₹80,000, excluding examination and other charges. For M.Tech, the annual fee is around ₹60,000. Ph.D. scholars pay a nominal registration and semester fee. Fees may vary slightly each academic year and are subject to university revisions.",
        metadata={"sources": "fees.txt"}
    ),
    Document(
        page_content="USICT offers integrated B.Tech + M.Tech dual degree programs in CSE, IT, ECE, and BT. It also provides standalone M.Tech degrees in niche areas like Robotics & Automation and Signal Processing. Ph.D. programs in related domains are active and well-supported by research labs.",
        metadata={"sources":"courses.txt"}
    ),
    Document(
        page_content="USICT boasts strong placement records with companies like Infosys, TCS, and Nagarro. The school collaborates with industry through internships, live projects, and technical workshops to ensure students are industry-ready.",
        metadata={"sources":"placement.txt"}
    )
]

db = Chroma.from_documents(docs, embedding)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3})
# res = retriever.invoke("what do I have to pay?")
# print(res)

template = """
Answer the question based on this context: {content}
\n\n\n question {question}
"""
prompt= ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm