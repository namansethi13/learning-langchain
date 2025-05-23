from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import SystemMessage, HumanMessage


def generate_blog(subject):

    titlePrompt = PromptTemplate.from_template(
        "Generate a title for a blog about {subject}. Just a single title, no other text."
    )
    contentPrompt = PromptTemplate.from_template(
        "Generate a detailed blog post with the title {title}. The blog should be informative, engaging, and suitable for a general audience, keep it under 500 words. Just the blog content, no other text."
    )
    
    titleChain = titlePrompt | ChatGoogleGenerativeAI(temperature=0.8, model="gemini-2.0-flash")
    contentChain = contentPrompt | ChatGoogleGenerativeAI(temperature=0.6, model="gemini-2.0-flash")

    blogChain = RunnableSequence(
        lambda inputs: {"title": titleChain.invoke({"subject": inputs["subject"]}).content},
        lambda inputs: {
            "title": inputs["title"],
            "content": contentChain.invoke({"title": inputs["title"]}).content
        }
    )


    
    result = blogChain.invoke({"subject": subject})
    blogTitle = result["title"]
    blogContent = result["content"]
    return blogTitle, blogContent





if __name__ == "__main__":
    subject = input("Enter the subject of the blog you want to generate: ")
    blogTitle, blogContent = generate_blog(subject)
    print(f"Blog Title: {blogTitle}")
    print(f"Blog Content: {blogContent}")



