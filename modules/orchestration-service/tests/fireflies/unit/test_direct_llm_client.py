#!/usr/bin/env python3
"""
LLMClient Unit Tests

Tests the unified LLMClient and conversation helper functions with REAL logic:
- Client construction with various configs (direct and proxy modes)
- Circuit breaker integration
- _build_conversation_messages() context window management
- _flatten_messages_to_prompt() format
- _generate_suggested_queries() heuristics
- Config defaults for direct LLM settings

Uses real PostgreSQL database (no SQLite, no mocks).
All test output goes to tests/output/ with timestamp format.

Shared DB fixtures (db_session_factory, bot_session_id, intelligence_service)
are provided by tests/fireflies/conftest.py using real PostgreSQL.
"""

import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

# Output directory
OUTPUT_DIR = orchestration_root / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_output_path(test_name: str) -> Path:
    """Get timestamped output file path."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_test_{test_name}_results.log"


def write_output(path: Path, content: str):
    """Write test output to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# LLMClient Construction Tests
# =============================================================================


class TestLLMClientConstruction:
    """Test LLMClient initialization with various configs."""

    def test_default_construction(self):
        """Test default client construction (direct mode)."""
        from clients.llm_client import LLMClient

        client = LLMClient()
        assert client.base_url == "http://localhost:11434/v1"
        assert client.api_key == ""
        assert client.model == "gemma3:4b"
        assert client.timeout == 120.0
        assert client.default_max_tokens == 1024
        assert client.default_temperature == 0.3
        assert client.proxy_mode is False

        output = get_output_path("llm_client_default_construction")
        write_output(output, "LLMClient default construction OK (direct mode)")

    def test_custom_construction(self):
        """Test client with custom parameters."""
        from clients.llm_client import LLMClient

        client = LLMClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test-123",  # pragma: allowlist secret
            model="gpt-4o",
            timeout=60.0,
            default_max_tokens=2048,
            default_temperature=0.7,
        )
        assert client.base_url == "https://api.openai.com/v1"
        assert client.api_key == "sk-test-123"  # pragma: allowlist secret
        assert client.model == "gpt-4o"
        assert client.timeout == 60.0
        assert client.default_max_tokens == 2048
        assert client.default_temperature == 0.7

    def test_proxy_mode_construction(self):
        """Test client construction in proxy mode."""
        from clients.llm_client import LLMClient

        client = LLMClient(
            base_url="http://localhost:5003",
            proxy_mode=True,
            default_backend="ollama",
        )
        assert client.base_url == "http://localhost:5003"
        assert client.proxy_mode is True
        assert client.default_backend == "ollama"

    def test_base_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base_url."""
        from clients.llm_client import LLMClient

        client = LLMClient(base_url="http://localhost:11434/v1/")
        assert client.base_url == "http://localhost:11434/v1"

    def test_factory_function(self):
        """Test the create_llm_client factory."""
        from clients.llm_client import create_llm_client

        client = create_llm_client(
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3",
            timeout=90.0,
        )
        assert client.model == "llama3"
        assert client.timeout == 90.0

    def test_factory_function_proxy_mode(self):
        """Test the create_llm_client factory with proxy mode."""
        from clients.llm_client import create_llm_client

        client = create_llm_client(
            base_url="http://localhost:5003",
            proxy_mode=True,
            default_backend="groq",
        )
        assert client.proxy_mode is True
        assert client.default_backend == "groq"

    def test_headers_without_api_key(self):
        """Test headers when no API key is set."""
        from clients.llm_client import LLMClient

        client = LLMClient(api_key="")
        headers = client._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_with_api_key(self):
        """Test headers when API key is set."""
        from clients.llm_client import LLMClient

        client = LLMClient(api_key="sk-test-key")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer sk-test-key"

    def test_has_all_methods(self):
        """Test that LLMClient has all expected methods."""
        from clients.llm_client import LLMClient

        client = LLMClient()
        assert hasattr(client, "chat")
        assert hasattr(client, "chat_stream")
        assert hasattr(client, "translate_prompt")
        assert hasattr(client, "translate_prompt_stream")
        assert hasattr(client, "health_check")
        assert hasattr(client, "connect")
        assert hasattr(client, "close")

        output = get_output_path("llm_client_interface")
        write_output(
            output,
            "LLMClient interface validated:\n"
            "  - chat()\n"
            "  - chat_stream()\n"
            "  - translate_prompt()\n"
            "  - translate_prompt_stream()\n"
            "  - health_check()\n"
            "  - connect()\n"
            "  - close()\n",
        )

    def test_satisfies_protocol(self):
        """Test that LLMClient satisfies LLMClientProtocol."""
        from clients.llm_client import LLMClient
        from clients.protocol import LLMClientProtocol

        client = LLMClient()
        assert isinstance(client, LLMClientProtocol)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakerIntegration:
    """Test circuit breaker in LLMClient."""

    def test_circuit_breaker_initial_state(self):
        """Test that circuit breaker starts closed."""
        from clients.llm_client import LLMClient

        client = LLMClient()
        assert client._circuit_breaker.state == "closed"
        assert client._circuit_breaker.is_available is True

    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        from clients.llm_client import LLMClient

        client = LLMClient()
        for _ in range(5):
            client._circuit_breaker.record_failure()

        assert client._circuit_breaker.state == "open"
        assert client._circuit_breaker.is_available is False

    def test_circuit_breaker_resets_on_success(self):
        """Test that circuit breaker resets on success."""
        from clients.llm_client import LLMClient

        client = LLMClient()
        # Record some failures (not enough to open)
        for _ in range(3):
            client._circuit_breaker.record_failure()

        client._circuit_breaker.record_success()
        assert client._circuit_breaker.state == "closed"
        assert client._circuit_breaker._failure_count == 0

        output = get_output_path("llm_client_circuit_breaker")
        write_output(output, "Circuit breaker integration tests passed")


# =============================================================================
# _build_conversation_messages() Tests
# =============================================================================


class TestBuildConversationMessages:
    """Test the context window management function."""

    def test_basic_message_building(self):
        """Test building messages with system context and user message."""
        from services.meeting_intelligence import _build_conversation_messages

        messages = _build_conversation_messages(
            system_context="You are a meeting analyst.",
            message_history=[],
            new_user_message="What were the action items?",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a meeting analyst."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What were the action items?"

        output = get_output_path("build_messages_basic")
        write_output(output, f"Basic messages built: {len(messages)} messages")

    def test_no_system_context(self):
        """Test building messages without system context."""
        from services.meeting_intelligence import _build_conversation_messages

        messages = _build_conversation_messages(
            system_context=None,
            message_history=[],
            new_user_message="Hello",
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_with_history(self, db_session_factory, bot_session_id):
        """Test building messages with real AgentMessage records from DB."""
        from database.models import AgentConversation, AgentMessage
        from services.meeting_intelligence import _build_conversation_messages

        cid = uuid.uuid4()

        # Create real conversation and messages in DB
        async with db_session_factory() as session:
            conv = AgentConversation(
                conversation_id=cid,
                session_id=uuid.UUID(bot_session_id),
                title="History Build Test",
                status="active",
            )
            session.add(conv)

            msg1 = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="user",
                content="What happened in the meeting?",
            )
            msg2 = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="assistant",
                content="The team discussed the roadmap.",
            )
            session.add_all([msg1, msg2])
            await session.commit()

            # Fetch real ORM objects
            from sqlalchemy import select

            result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == cid)
                .order_by(AgentMessage.created_at.asc())
            )
            history = list(result.scalars().all())

        messages = _build_conversation_messages(
            system_context="You are an assistant.",
            message_history=history,
            new_user_message="What were the action items?",
        )

        # system + 2 history + new user = 4
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What happened in the meeting?"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "What were the action items?"

    @pytest.mark.asyncio
    async def test_context_window_truncation(self, db_session_factory, bot_session_id):
        """Test that old messages are dropped when context window is exceeded (real DB)."""
        from database.models import AgentConversation, AgentMessage
        from services.meeting_intelligence import _build_conversation_messages

        cid = uuid.uuid4()

        # Create real conversation with 50 messages in DB
        async with db_session_factory() as session:
            conv = AgentConversation(
                conversation_id=cid,
                session_id=uuid.UUID(bot_session_id),
                title="Truncation Test",
                status="active",
            )
            session.add(conv)

            for i in range(50):
                msg = AgentMessage(
                    message_id=uuid.uuid4(),
                    conversation_id=cid,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}: " + "x" * 200,  # ~50 tokens each
                )
                session.add(msg)
            await session.commit()

            # Fetch real ORM objects
            from sqlalchemy import select

            result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == cid)
                .order_by(AgentMessage.created_at.asc())
            )
            history = list(result.scalars().all())

        assert len(history) == 50  # Verify real records created

        messages = _build_conversation_messages(
            system_context="System prompt " + "y" * 200,
            message_history=history,
            new_user_message="Latest question?",
            max_tokens=512,  # Very small budget
        )

        # Should have fewer messages than history + 2
        assert len(messages) < 52
        # System is always first
        assert messages[0]["role"] == "system"
        # New user message is always last
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Latest question?"

        output = get_output_path("context_window_truncation")
        write_output(
            output,
            f"Context truncation: 50 real DB messages -> {len(messages) - 2} kept "
            f"(max_tokens=512)\n",
        )

    def test_empty_history_and_context(self):
        """Test with no history and no system context."""
        from services.meeting_intelligence import _build_conversation_messages

        messages = _build_conversation_messages(
            system_context=None,
            message_history=[],
            new_user_message="Just a question",
        )

        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Just a question"}


# =============================================================================
# _flatten_messages_to_prompt() Tests
# =============================================================================


class TestFlattenMessagesToPrompt:
    """Test the message flattening for proxy mode LLM calls."""

    def test_basic_flatten(self):
        """Test flattening messages to prompt string."""
        from services.meeting_intelligence import _flatten_messages_to_prompt

        messages = [
            {"role": "system", "content": "You are a meeting analyst."},
            {"role": "user", "content": "What happened?"},
        ]

        prompt, system_prompt = _flatten_messages_to_prompt(messages)

        assert system_prompt == "You are a meeting analyst."
        assert "USER: What happened?" in prompt

        output = get_output_path("flatten_messages_basic")
        write_output(
            output,
            f"Flattened messages:\n  system_prompt: {system_prompt[:50]}...\n  prompt: {prompt}\n",
        )

    def test_multi_turn_flatten(self):
        """Test flattening multi-turn conversation."""
        from services.meeting_intelligence import _flatten_messages_to_prompt

        messages = [
            {"role": "system", "content": "System prompt here."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ]

        prompt, system_prompt = _flatten_messages_to_prompt(messages)

        assert system_prompt == "System prompt here."
        assert "USER: Question 1" in prompt
        assert "ASSISTANT: Answer 1" in prompt
        assert "USER: Question 2" in prompt

    def test_no_system_message(self):
        """Test flattening without system message."""
        from services.meeting_intelligence import _flatten_messages_to_prompt

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        prompt, system_prompt = _flatten_messages_to_prompt(messages)

        assert system_prompt == ""
        assert prompt == "USER: Hello"


# =============================================================================
# _generate_suggested_queries() Tests
# =============================================================================


class TestGenerateSuggestedQueries:
    """Test the heuristic suggestion generator."""

    def test_base_suggestions_always_present(self):
        """Test that base suggestions are always returned."""
        from services.meeting_intelligence import _generate_suggested_queries

        queries = _generate_suggested_queries("Some random text.", "Meeting Q&A")
        assert len(queries) >= 2
        assert "Can you elaborate on that?" in queries
        assert "What are the next steps?" in queries

    def test_action_item_suggestions(self):
        """Test suggestions when response mentions action items."""
        from services.meeting_intelligence import _generate_suggested_queries

        queries = _generate_suggested_queries(
            "The main action items from the meeting are: update the docs, review the PR.",
            "Meeting Q&A",
        )
        assert any("responsible" in q.lower() or "deadline" in q.lower() for q in queries)

    def test_decision_suggestions(self):
        """Test suggestions when response mentions decisions."""
        from services.meeting_intelligence import _generate_suggested_queries

        queries = _generate_suggested_queries(
            "The team decided to use React for the frontend.",
            "Meeting Q&A",
        )
        assert any("alternatives" in q.lower() or "risk" in q.lower() for q in queries)

    def test_risk_suggestions(self):
        """Test suggestions when response mentions risks."""
        from services.meeting_intelligence import _generate_suggested_queries

        queries = _generate_suggested_queries(
            "There are concerns about the timeline and potential issues.",
            "Meeting Q&A",
        )
        assert any("mitigate" in q.lower() or "priority" in q.lower() for q in queries)

    def test_max_five_suggestions(self):
        """Test that at most 5 suggestions are returned."""
        from services.meeting_intelligence import _generate_suggested_queries

        # Trigger all keyword categories
        queries = _generate_suggested_queries(
            "The decision about the budget risk involves action items and timeline concerns for each speaker.",
            "Meeting Q&A",
        )
        assert len(queries) <= 5

    def test_no_duplicates(self):
        """Test that suggestions contain no duplicates."""
        from services.meeting_intelligence import _generate_suggested_queries

        queries = _generate_suggested_queries("Action items and tasks to assign.", "Meeting Q&A")
        assert len(queries) == len(set(queries))

        output = get_output_path("suggested_queries_heuristic")
        write_output(
            output,
            f"Suggested queries tests passed:\n  Sample: {queries}\n",
        )


# =============================================================================
# Config Tests for Direct LLM Settings
# =============================================================================


class TestDirectLLMConfig:
    """Test direct LLM configuration fields."""

    def test_direct_llm_settings_fields(self):
        """Test that direct LLM settings fields exist with valid types."""
        from config import MeetingIntelligenceSettings

        settings = MeetingIntelligenceSettings()
        assert isinstance(settings.direct_llm_enabled, bool)
        assert isinstance(settings.direct_llm_base_url, str)
        assert len(settings.direct_llm_base_url) > 0
        assert isinstance(settings.direct_llm_api_key, str)
        assert isinstance(settings.direct_llm_model, str)
        assert len(settings.direct_llm_model) > 0
        assert isinstance(settings.agent_max_context_tokens, int)
        assert settings.agent_max_context_tokens > 0

        output = get_output_path("direct_llm_config_fields")
        write_output(
            output,
            f"Direct LLM config fields:\n"
            f"  enabled: {settings.direct_llm_enabled}\n"
            f"  base_url: {settings.direct_llm_base_url}\n"
            f"  model: {settings.direct_llm_model}\n"
            f"  max_context_tokens: {settings.agent_max_context_tokens}\n",
        )

    def test_direct_llm_in_main_settings(self):
        """Test that direct LLM settings are accessible from main Settings."""
        from config import Settings

        settings = Settings()
        intel = settings.intelligence
        assert hasattr(intel, "direct_llm_enabled")
        assert hasattr(intel, "direct_llm_base_url")
        assert hasattr(intel, "direct_llm_api_key")
        assert hasattr(intel, "direct_llm_model")
        assert hasattr(intel, "agent_max_context_tokens")


# =============================================================================
# Real DB: Agent Chat with No LLM (Graceful Fallback)
# =============================================================================


class TestAgentChatNoLLMRealDB:
    """Test agent chat behavior when no LLM client is available."""

    @pytest.fixture
    async def intelligence_service_no_llm(self, db_session_factory):
        """Create service with no translation client."""
        from config import MeetingIntelligenceSettings
        from services.meeting_intelligence import MeetingIntelligenceService

        return MeetingIntelligenceService(
            db_session_factory=db_session_factory,
            translation_client=None,
            settings=MeetingIntelligenceSettings(),
        )

    @pytest.mark.asyncio
    async def test_send_message_without_llm_returns_helpful_error(
        self, intelligence_service_no_llm, db_session_factory, bot_session_id
    ):
        """Test that sending a message without LLM returns a helpful message."""
        service = intelligence_service_no_llm

        conv = await service.create_conversation(
            session_id=bot_session_id,
            title="No LLM Test",
        )

        reply = await service.send_message(
            conversation_id=conv["conversation_id"],
            content="What were the action items?",
        )

        assert reply["role"] == "assistant"
        assert "No LLM backend" in reply["content"]
        assert reply["suggested_queries"] is not None
        assert len(reply["suggested_queries"]) > 0
        # Should NOT have stub metadata
        assert reply.get("message_metadata", {}).get("stub") is not True

        output = get_output_path("agent_chat_no_llm")
        write_output(
            output,
            f"Agent chat without LLM:\n"
            f"  reply: {reply['content'][:100]}...\n"
            f"  has suggestions: {len(reply['suggested_queries'])}\n",
        )

    @pytest.mark.asyncio
    async def test_conversation_not_found_raises_error(self, intelligence_service_no_llm):
        """Test that sending message to non-existent conversation raises ValueError."""
        service = intelligence_service_no_llm
        fake_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="not found"):
            await service.send_message(
                conversation_id=fake_id,
                content="Hello?",
            )

    @pytest.mark.asyncio
    async def test_message_history_persists_in_db(
        self, intelligence_service_no_llm, db_session_factory, bot_session_id
    ):
        """Test that both user and assistant messages are persisted."""
        service = intelligence_service_no_llm

        conv = await service.create_conversation(
            session_id=bot_session_id,
            title="History Test",
        )

        # Send two messages
        await service.send_message(conv["conversation_id"], "Question 1")
        await service.send_message(conv["conversation_id"], "Question 2")

        history = await service.get_conversation_history(conv["conversation_id"])
        assert len(history["messages"]) == 4  # 2 user + 2 assistant

        roles = [m["role"] for m in history["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]

        output = get_output_path("agent_chat_history_persistence")
        write_output(
            output,
            f"Message history persisted: {len(history['messages'])} messages\n"
            f"  roles: {roles}\n",
        )


# =============================================================================
# Streaming SSE Endpoint Structure Tests
# =============================================================================


class TestStreamingEndpointStructure:
    """Test that the streaming endpoint is properly registered."""

    def test_streaming_endpoint_exists(self):
        """Test that the streaming SSE endpoint is registered."""
        from routers.insights import router

        paths = set()
        for route in router.routes:
            path = getattr(route, "path", "")
            if path:
                paths.add(path)

        assert "/agent/conversations/{conversation_id}/messages/stream" in paths

        output = get_output_path("streaming_endpoint_structure")
        write_output(
            output,
            "Streaming endpoint registered:\n"
            "  /agent/conversations/{conversation_id}/messages/stream\n",
        )

    def test_streaming_endpoint_is_post(self):
        """Test that the streaming endpoint uses POST method."""
        from routers.insights import router

        for route in router.routes:
            path = getattr(route, "path", "")
            if path == "/agent/conversations/{conversation_id}/messages/stream":
                methods = getattr(route, "methods", set())
                assert "POST" in methods
                break
