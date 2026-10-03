import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # silence ChromaDB telemetry errors

# Any OpenAI-compatible provider works (Groq, OpenRouter, Ollama, DeepInfra...).
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")


def get_llm_client():
    """Return an OpenAI-compatible client configured from the .env file."""
    if not LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is missing: copy .env.example to .env and fill in your API key."
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
