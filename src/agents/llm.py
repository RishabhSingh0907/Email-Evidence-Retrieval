"""
src/agents/llm.py
-----------------
LLM singleton for the Email Evidence Agent.

Model  : llama-3.3-70b-versatile via Groq (free tier)
Reason : Groq is free, fast (~300 tok/s), supports tool-calling natively,
         and llama-3.3-70b is the strongest free model available as of 2025.

Fallback order (all free on Groq):
  1. llama-3.3-70b-versatile   — best reasoning, tool-calling
  2. llama-3.1-70b-versatile   — older variant, same capability tier
  3. llama-3.1-8b-instant      — fast, lower quality

.env key required:
  GROQ_API_KEY=gsk_...
"""

import openai
from openai import OpenAI
import logging
from langchain_openai import ChatOpenAI
import dotenv
import os
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

_GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_TEMP       = float(os.getenv("LLM_TEMPERATURE", "0.0"))   # 0 = deterministic, best for evidence

# Create the LLM
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",  # Specify the Groq model ID
    api_key=_GROQ_KEY,
    temperature=_TEMP,
    base_url="https://api.groq.com/openai/v1",  # Groq's OpenAI-compatible endpoint
)