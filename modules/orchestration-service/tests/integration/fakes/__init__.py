"""In-process LLM fake servers for integration tests.

Real aiohttp servers — not mocks. Bound to localhost on an OS-assigned port
so multiple tests can run in parallel without port conflicts. Each instance
records every request it sees so tests can assert on URLs, headers, and
request bodies post-hoc.
"""

from .fake_anthropic_server import FakeAnthropicServer
from .fake_ollama_server import FakeOllamaServer
from .fake_openai_server import FakeOpenAIServer

__all__ = [
    "FakeAnthropicServer",
    "FakeOllamaServer",
    "FakeOpenAIServer",
]
