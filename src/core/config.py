
"""
Core configuration and environment settings.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Groq LLM configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    # Gemini configuration (embeddings / fallback)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "models/gemini-embedding-001"
    )

    # Tavily configuration
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    CODE_COLLECTION = os.getenv("QDRANT_CODE_COLLECTION", "codebase")
    DOCS_COLLECTION = os.getenv("QDRANT_DOCS_COLLECTION", "guidelines")


settings = Settings()

# Set environment variables for LangChain integrations
os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
