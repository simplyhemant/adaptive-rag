
"""
Groq LLM initialization and configuration.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.core.config import settings

load_dotenv()

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0
)
