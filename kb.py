# -*- coding: utf-8 -*-
"""
Knowledge Base Integration for Presales Voicebot
- Azure AI Search with vector search
- Async/await for low latency
- Intelligent caching to avoid duplicate searches
- Text sanitization
"""
import asyncio
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential
from config import config
from colorama import Fore, Style


class SearchCache:
    """Simple in-memory cache with TTL for search results"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[List[Dict], datetime]] = {}
        self.ttl_seconds = ttl_seconds

    def _generate_key(self, query: str, top_k: int) -> str:
        """Generate cache key from query and top_k"""
        key_string = f"{query.lower().strip()}|{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, query: str, top_k: int) -> Optional[List[Dict]]:
        """Get cached results if available and not expired"""
        key = self._generate_key(query, top_k)

        if key in self.cache:
            results, timestamp = self.cache[key]
            # Check if cache is still valid
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return results
            else:
                # Remove expired entry
                del self.cache[key]

        return None

    def set(self, query: str, top_k: int, results: List[Dict]):
        """Store results in cache"""
        key = self._generate_key(query, top_k)
        self.cache[key] = (results, datetime.now())

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()


class KnowledgeBase:
    """
    Knowledge base search using Azure AI Search
    - Vector search with integrated vectorization
    - Async operations for low latency
    - Intelligent caching
    - Text sanitization
    """

    def __init__(self):
        self.config = config
        self.search_client: Optional[SearchClient] = None
        self.cache = SearchCache(ttl_seconds=config.cache_ttl_seconds) if config.enable_search_cache else None
        self.vector_field_name = config.vector_field_name

    def initialize(self) -> bool:
        """Initialize Azure AI Search client"""
        try:
            if not config.enable_knowledge_base:
                print(f"{Fore.YELLOW}Knowledge base is disabled in configuration{Style.RESET_ALL}")
                return False

            self.search_client = SearchClient(
                endpoint=config.azure_search_endpoint,
                index_name=config.azure_search_index,
                credential=AzureKeyCredential(config.azure_search_api_key)
            )

            print(f"{Fore.GREEN} Azure AI Search initialized")
            print(f"{Fore.CYAN}  Index: {config.azure_search_index}")
            print(f"{Fore.CYAN}  Vector Field: {self.vector_field_name}")
            print(f"{Fore.CYAN}  Cache: {'Enabled' if self.cache else 'Disabled'}{Style.RESET_ALL}")
            return True

        except Exception as e:
            print(f"{Fore.RED}Failed to initialize Azure AI Search: {e}{Style.RESET_ALL}")
            return False

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove control characters and sanitize text"""
        if not text:
            return ""

        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())

        return sanitized.strip()

    async def search_async(self, query_text: str, top_k: int = None) -> Tuple[List[Dict], float]:
        """
        Async wrapper for Azure AI Search vector search
        Returns: (documents, latency_ms)
        """
        if top_k is None:
            top_k = config.search_top_k

        # Check cache first
        if self.cache:
            cached_results = self.cache.get(query_text, top_k)
            if cached_results is not None:
                print(f"{Fore.MAGENTA} Cache hit for query{Style.RESET_ALL}")
                return cached_results, 0.0  # Cached, so latency is negligible

        # Run search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results, latency = await loop.run_in_executor(
            None,
            self._search_sync,
            query_text,
            top_k
        )

        # Store in cache
        if self.cache:
            self.cache.set(query_text, top_k, results)

        return results, latency

    def _search_sync(self, query_text: str, top_k: int) -> Tuple[List[Dict], float]:
        """
        Synchronous vector search using Azure AI Search
        - Pure vector similarity search
        - Azure AI Search handles embedding generation automatically
        """
        if not self.search_client:
            return [], 0.0

        start_time = time.perf_counter()

        try:
            # Use VectorizableTextQuery - Azure AI Search vectorizes automatically
            vector_query = VectorizableTextQuery(
                text=query_text,
                k_nearest_neighbors=top_k,
                fields=self.vector_field_name
            )

            # Pure vector search (no keyword search)
            results = self.search_client.search(
                search_text=None,  # None = pure vector search
                vector_queries=[vector_query],
                select=["title", "chunk"],  # Only fetch needed fields
                top=top_k
            )

            # Process results
            docs = []
            seen_chunks = set()  # For deduplication

            for r in results:
                chunk = self.sanitize_text(r.get("chunk", ""))

                # Deduplicate by chunk content
                if chunk and chunk not in seen_chunks:
                    docs.append({
                        "title": self.sanitize_text(r.get("title", "N/A")),
                        "chunk": chunk,
                        "score": r.get("@search.score", 0)
                    })
                    seen_chunks.add(chunk)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            return docs, latency_ms

        except Exception as e:
            print(f"{Fore.RED}Search error: {e}{Style.RESET_ALL}")
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            return [], latency_ms

    def format_context_for_prompt(self, docs: List[Dict]) -> str:
        """
        Format search results into context string for the LLM
        """
        if not docs:
            return ""

        context_parts = ["KNOWLEDGE BASE CONTEXT:"]

        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "N/A")
            chunk = doc.get("chunk", "")
            score = doc.get("score", 0)

            context_parts.append(f"\n[Source {i}] {title} (relevance: {score:.2f})")
            context_parts.append(chunk)

        context_parts.append("\n---\nUse the above context to answer the user's question. If the context doesn't contain relevant information, say so politely.")

        return "\n".join(context_parts)


# Global knowledge base instance
knowledge_base = KnowledgeBase()
