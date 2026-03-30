from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import model_garden_maas


class ModelFamily(Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LLAMA = "llama"


@dataclass
class ModelConfig:
    family: ModelFamily
    name: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class ModelSelector:
    def __init__(self):
        self._api_keys = {
            ModelFamily.GPT: os.getenv("OPENAI_API_KEY"),
            ModelFamily.CLAUDE: os.getenv("ANTHROPIC_API_KEY"),
            ModelFamily.GEMINI: os.getenv("GEMINI_API_KEY"),
            #ModelFamily.LLAMA: os.getenv("GOOGLE_API_KEY"),
        }
        self._vertex_project = os.getenv("VERTEX_PROJECT_ID")

        for family, key in self._api_keys.items():
            if not key:
                print(f"Warning: No API key found for {family.value}")
        if self._vertex_project is None:
            print("Warning: No Vertex project ID set (VERTEX_PROJECT_ID)")

    def get_model(self, config: ModelConfig) -> BaseChatModel:
        # Special handling for LLAMA/VertexAI - it doesn't need an API key
        if config.family == ModelFamily.LLAMA:
            if self._vertex_project is None:
                raise ValueError(
                    "VERTEX_PROJECT_ID must be set to use VertexAI models")
            return model_garden_maas.VertexModelGardenLlama(
                model_name=config.name,
                project=self._vertex_project,
                location="us-east5",
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        # For all other models, check API key
        api_key = self._api_keys.get(config.family)
        if not api_key:
            raise ValueError(f"No API key available for {config.family.value}")

        if config.family == ModelFamily.GPT:
            return ChatOpenAI(
                model=config.name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        elif config.family == ModelFamily.CLAUDE:
            return ChatAnthropic(
                model=config.name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        elif config.family == ModelFamily.GEMINI:
            return ChatGoogleGenerativeAI(
                model=config.name,
                google_api_key=api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens
            )
    
        else:
            raise ValueError(f"Unsupported model family: {config.family}")

    def is_available(self, family: ModelFamily) -> bool:
        if family == ModelFamily.LLAMA:
            return bool(self._api_keys.get(family) and self._vertex_project)
        return bool(self._api_keys.get(family))
