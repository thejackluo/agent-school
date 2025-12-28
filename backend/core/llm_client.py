"""
LLM Client for interacting with OpenAI and Anthropic APIs.

This module provides a unified interface for generating text using different LLM providers
with retry logic, caching, and cost tracking.
"""

import os
import asyncio
import hashlib
import json
import logging
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMClient:
    """
    Unified client for interacting with LLM providers.

    Supports OpenAI and Anthropic APIs with automatic retry, caching,
    and cost tracking.

    Examples:
        >>> client = LLMClient(provider=LLMProvider.OPENAI)
        >>> response = await client.generate(
        ...     prompt="Generate a plan for searching events",
        ...     system="You are a workflow planning expert"
        ... )
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider to use (OpenAI or Anthropic)
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
        """
        self.provider = provider
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._total_cost = 0.0
        self._request_count = 0

        # Initialize provider-specific client
        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
            self.cost_per_1k_input = 0.01  # $0.01 per 1K input tokens
            self.cost_per_1k_output = 0.03  # $0.03 per 1K output tokens
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = AsyncAnthropic(api_key=api_key)
            self.model = "claude-3-opus-20240229"
            self.cost_per_1k_input = 0.015  # $0.015 per 1K input tokens
            self.cost_per_1k_output = 0.075  # $0.075 per 1K output tokens

        logger.info(f"Initialized LLM client with provider: {provider.value}, model: {self.model}")

    def _get_cache_key(self, prompt: str, system: Optional[str], temperature: float) -> str:
        """Generate cache key for request."""
        content = f"{prompt}|{system}|{temperature}|{self.model}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Retrieve response from cache if not expired."""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if datetime.now() < entry["expires_at"]:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return entry["response"]
            else:
                # Remove expired entry
                del self._cache[cache_key]
                logger.debug(f"Cache expired for key: {cache_key[:8]}...")
        return None

    def _add_to_cache(self, cache_key: str, response: str):
        """Add response to cache."""
        self._cache[cache_key] = {
            "response": response,
            "expires_at": datetime.now() + self.cache_ttl
        }
        logger.debug(f"Added to cache: {cache_key[:8]}...")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        use_cache: bool = True,
        retry_attempts: int = 3
    ) -> str:
        """
        Generate text using the configured LLM provider.

        Args:
            prompt: User prompt/query
            system: System message to set context
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use response cache
            retry_attempts: Number of retry attempts on failure

        Returns:
            Generated text response

        Raises:
            Exception: If all retry attempts fail
        """
        # Check cache
        cache_key = self._get_cache_key(prompt, system, temperature)
        if use_cache:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response

        # Attempt generation with retries
        last_error = None
        for attempt in range(retry_attempts):
            try:
                response = await self._generate_with_provider(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Track metrics
                self._request_count += 1

                # Cache successful response
                if use_cache:
                    self._add_to_cache(cache_key, response)

                return response

            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Attempt {attempt + 1}/{retry_attempts} failed: {str(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(wait_time)

        # All retries failed
        logger.error(f"All {retry_attempts} attempts failed. Last error: {str(last_error)}")
        raise last_error

    async def _generate_with_provider(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate text using the specific provider."""
        if self.provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, system, temperature, max_tokens)
        else:
            return await self._generate_anthropic(prompt, system, temperature, max_tokens)

    async def _generate_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate text using OpenAI API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Track costs
        usage = response.usage
        input_cost = (usage.prompt_tokens / 1000) * self.cost_per_1k_input
        output_cost = (usage.completion_tokens / 1000) * self.cost_per_1k_output
        self._total_cost += input_cost + output_cost

        logger.info(
            f"OpenAI request complete. Tokens: {usage.prompt_tokens} in, "
            f"{usage.completion_tokens} out. Cost: ${input_cost + output_cost:.4f}"
        )

        return response.choices[0].message.content

    async def _generate_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate text using Anthropic API."""
        response = await self.client.messages.create(
            model=self.model,
            system=system or "You are a helpful AI assistant.",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Track costs (approximate - Anthropic doesn't always return exact token counts)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        self._total_cost += input_cost + output_cost

        logger.info(
            f"Anthropic request complete. Tokens: {input_tokens} in, "
            f"{output_tokens} out. Cost: ${input_cost + output_cost:.4f}"
        )

        return response.content[0].text

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with request count, total cost, and cache stats
        """
        return {
            "provider": self.provider.value,
            "model": self.model,
            "total_requests": self._request_count,
            "total_cost_usd": round(self._total_cost, 4),
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self.cache_ttl.total_seconds()
        }

    def clear_cache(self):
        """Clear the response cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared cache ({cache_size} entries)")


# Example usage
async def main():
    """Example usage of LLMClient."""
    client = LLMClient(provider=LLMProvider.OPENAI)

    response = await client.generate(
        prompt="Write a brief plan for extracting events from a website.",
        system="You are a workflow planning expert.",
        temperature=0.7
    )

    print("Response:", response)
    print("\nStats:", json.dumps(client.get_stats(), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
