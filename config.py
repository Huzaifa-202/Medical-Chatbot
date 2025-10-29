# -*- coding: utf-8 -*-
"""
Configuration management for Presales Voicebot
Loads settings from environment variables
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for voicebot and knowledge base"""

    def __init__(self):
        # Azure OpenAI Configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-realtime-preview")

        # Azure AI Search Configuration
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
        self.vector_field_name = os.getenv("VECTOR_FIELD_NAME", "text_vector")

        # Voice settings
        self.voice = os.getenv("VOICE", "alloy")

        # Audio settings
        self.sample_rate = 24000
        self.chunk_size = 4096

        # VAD settings
        self.enable_vad = os.getenv("ENABLE_VAD", "true").lower() == "true"

        # Knowledge base settings
        self.enable_knowledge_base = os.getenv("ENABLE_KNOWLEDGE_BASE", "true").lower() == "true"
        self.search_top_k = int(os.getenv("SEARCH_TOP_K", "3"))
        self.enable_search_cache = os.getenv("ENABLE_SEARCH_CACHE", "true").lower() == "true"
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes default
        self.print_retrieved_docs = os.getenv("PRINT_RETRIEVED_DOCS", "true").lower() == "true"

    def validate(self) -> bool:
        """Validate that all required configuration is present"""
        required_fields = {
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
        }

        # Check Azure Search only if knowledge base is enabled
        if self.enable_knowledge_base:
            required_fields.update({
                "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
                "AZURE_SEARCH_API_KEY": self.azure_search_api_key,
                "AZURE_SEARCH_INDEX": self.azure_search_index,
            })

        missing = [k for k, v in required_fields.items() if not v]

        if missing:
            print(f"Missing required environment variables: {', '.join(missing)}")
            return False

        return True


# Global config instance
config = Config()
