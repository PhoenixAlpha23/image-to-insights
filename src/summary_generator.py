import os
import requests
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class GroqLLM(LLM):
    model: str = "gemma2-9b-it"
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "custom_groq"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


# Prompt template
prompt = PromptTemplate(
    input_variables=["diff_keywords"],
    template="""
You are an assistant summarizing visual differences between two images.
Detected changes: {diff_keywords}

Write a clear and simple summary in 2-3 sentences.
"""
)

# Use in a Langchain chain
chain = LLMChain(llm=GroqLLM(), prompt=prompt)


def generate_summary(diff_keywords: list[str]) -> str:
    joined = ", ".join(diff_keywords)
    return chain.run(diff_keywords=joined)
