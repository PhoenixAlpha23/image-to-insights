from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import os

# Groq API setup (make sure your env var is set)
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0.3,
    model_name="gemma2-9b-it",
    api_key=groq_api_key
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["diff_keywords"],
    template="""
You are an assistant summarizing changes between two images of the same location.
Given the following detected changes:

{diff_keywords}

Write a 2-3 sentence summary in simple English.
"""
)

# Langchain chain
chain = LLMChain(llm=llm, prompt=prompt)

# Entry point
def generate_summary(diff_keywords: list[str]) -> str:
    joined = ", ".join(diff_keywords)
    result = chain.run(diff_keywords=joined)
    return result.strip()
