import os
import requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class GroqLLM(LLM):
    model: str = "gemma2-9b-it"
    temperature: float = 0.3
    
    @property
    def _llm_type(self) -> str: 
        return "custom_groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            if not os.getenv('GROQ_API_KEY'):
                return "Error: GROQ_API_KEY environment variable not set. Please set up your API key."
                
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
            
            if "choices" not in result or not result["choices"]:
                return f"API Error: {result.get('error', {}).get('message', 'Unknown error')}"
                
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"

prompt = PromptTemplate(
    input_variables=["diff_keywords"],
    template="""
You are an assistant summarizing visual differences between two images.
Detected changes: {diff_keywords}

Write a clear and specific summary of these changes in 1-2 short sentences.
If no changes were detected, just state that the images appear to be identical.
"""
)

chain = LLMChain(llm=GroqLLM(), prompt=prompt)

def generate_summary(diff_keywords: str) -> str:
    #no changes, return simple message
    if diff_keywords == "No significant changes detected between the images.":
        return "The images appear to be identical with no significant differences in detected objects."
    return chain.run(diff_keywords=diff_keywords)
