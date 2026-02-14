#!/usr/bin/env python3
"""
Real LLM Integration Tests

Tests the FULL pipeline with a real LLM backend (Ollama).
These tests actually call the LLM and verify real responses.

Marked with @pytest.mark.llm — automatically skipped if Ollama is not running.

Uses real PostgreSQL database (no SQLite, no mocks).
All test output goes to tests/output/ with timestamp format.

Shared DB fixtures (db_session_factory, bot_session_id, intelligence_service)
are provided by tests/fireflies/conftest.py using real PostgreSQL.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))

# Output directory
OUTPUT_DIR = orchestration_root / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:4b"  # Available on this machine


def get_output_path(test_name: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_test_{test_name}_results.log"


def write_output(path: Path, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# Skip if Ollama is not available
# =============================================================================


async def _ollama_available() -> bool:
    """Check if Ollama is running and has models."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{OLLAMA_BASE_URL}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    return OLLAMA_MODEL in models
                return False
    except Exception:
        return False


@pytest.fixture(scope="module")
def ollama_check():
    """Module-level check: skip all tests if Ollama is not available."""
    import asyncio

    loop = asyncio.new_event_loop()
    available = loop.run_until_complete(_ollama_available())
    loop.close()
    if not available:
        pytest.skip(
            f"Ollama not available at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}. "
            "Start Ollama and pull the model to run these tests."
        )


# =============================================================================
# LLM Client Fixture (real Ollama)
# =============================================================================


@pytest.fixture
async def llm_client(ollama_check):
    """Create a real LLMClient connected to Ollama."""
    from clients.llm_client import LLMClient

    client = LLMClient(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        timeout=120.0,
        default_max_tokens=256,
        default_temperature=0.3,
    )
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
async def intelligence_service_with_llm(db_session_factory, llm_client):
    """Create MeetingIntelligenceService wired to real Ollama."""
    from config import MeetingIntelligenceSettings
    from services.meeting_intelligence import MeetingIntelligenceService

    settings = MeetingIntelligenceSettings(
        direct_llm_enabled=True,
        direct_llm_base_url=OLLAMA_BASE_URL,
        direct_llm_model=OLLAMA_MODEL,
    )

    return MeetingIntelligenceService(
        db_session_factory=db_session_factory,
        translation_client=llm_client,
        settings=settings,
    )


# =============================================================================
# Sample transcript for tests
# =============================================================================

SAMPLE_TRANSCRIPT = """
Alice: Good morning everyone. Let's discuss the Q3 roadmap.
Bob: Sure. I think we should prioritize the mobile app launch.
Alice: Agreed. What's the timeline looking like?
Carol: Engineering estimates 6 weeks for the MVP. We need to finalize the API contracts by next Friday.
Bob: I can have the API spec ready by Wednesday. Carol, can you review it Thursday?
Carol: Yes, I'll block my Thursday afternoon for that.
Alice: Great. Let's also talk about the budget. We're $50K over on cloud costs.
Bob: We should migrate the staging environment to spot instances. That could save 60%.
Alice: Good idea. Bob, please create a ticket for that. Any other concerns?
Carol: We need to hire two more backend engineers. The current team is stretched thin.
Alice: I'll escalate that to HR today. Let's meet again next Monday to check progress.
""".strip()


# =============================================================================
# Test 1: LLMClient Health Check
# =============================================================================


class TestLLMClientHealth:
    """Test that LLMClient can reach Ollama."""

    @pytest.mark.asyncio
    async def test_health_check_passes(self, llm_client):
        """Verify Ollama is reachable and healthy."""
        healthy = await llm_client.health_check()
        assert healthy is True

        output = get_output_path("llm_health_check")
        write_output(output, f"LLM health check: healthy={healthy}\n")

    @pytest.mark.asyncio
    async def test_simple_chat_returns_text(self, llm_client):
        """Send a trivial message and verify we get text back."""
        result = await llm_client.chat(
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            max_tokens=32,
        )

        assert result.text is not None
        assert len(result.text.strip()) > 0
        assert result.processing_time_ms > 0
        assert result.backend_used == "direct_llm"
        assert result.model_used is not None

        output = get_output_path("llm_simple_chat")
        write_output(
            output,
            f"Simple chat response:\n"
            f"  text: {result.text}\n"
            f"  processing_time_ms: {result.processing_time_ms:.1f}\n"
            f"  model: {result.model_used}\n"
            f"  tokens: {result.tokens_used}\n",
        )


# =============================================================================
# Test 2: Agent Chat with Real LLM
# =============================================================================


class TestAgentChatRealLLM:
    """Test agent conversation with real Ollama responses."""

    @pytest.mark.asyncio
    async def test_agent_chat_answers_about_transcript(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Create a conversation with a transcript and ask a question. Verify real answer."""
        service = intelligence_service_with_llm

        # Seed templates
        await service.load_default_templates()

        # Create conversation with real transcript
        conv = await service.create_conversation(
            session_id=bot_session_id,
            title="Roadmap Discussion Q&A",
            transcript_text=SAMPLE_TRANSCRIPT,
        )

        assert conv["system_context"] is not None
        assert "Alice" in conv["system_context"]

        # Ask about action items
        reply = await service.send_message(
            conversation_id=conv["conversation_id"],
            content="What are the action items from this meeting?",
        )

        assert reply["role"] == "assistant"
        assert len(reply["content"]) > 20  # Real response, not an error stub
        assert reply["llm_backend"] == "direct_llm"
        assert reply["llm_model"] is not None
        assert reply["processing_time_ms"] > 0
        assert reply["suggested_queries"] is not None

        # The response should mention at least some action items from the transcript
        content_lower = reply["content"].lower()
        has_relevant_content = any(
            keyword in content_lower
            for keyword in [
                "api",
                "bob",
                "carol",
                "ticket",
                "hire",
                "hr",
                "budget",
                "spec",
                "mobile",
            ]
        )
        assert (
            has_relevant_content
        ), f"LLM response doesn't seem to reference the transcript. Got: {reply['content'][:200]}"

        output = get_output_path("agent_chat_real_llm")
        write_output(
            output,
            f"Agent chat with real LLM:\n"
            f"  question: What are the action items from this meeting?\n"
            f"  answer ({len(reply['content'])} chars): {reply['content']}\n"
            f"  llm_backend: {reply['llm_backend']}\n"
            f"  llm_model: {reply['llm_model']}\n"
            f"  processing_time_ms: {reply['processing_time_ms']:.1f}\n"
            f"  suggested_queries: {reply['suggested_queries']}\n",
        )

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Test multi-turn conversation where follow-up references prior context."""
        service = intelligence_service_with_llm

        conv = await service.create_conversation(
            session_id=bot_session_id,
            title="Multi-turn Test",
            transcript_text=SAMPLE_TRANSCRIPT,
        )

        # First message
        reply1 = await service.send_message(
            conversation_id=conv["conversation_id"],
            content="Who attended this meeting?",
        )
        assert reply1["llm_backend"] == "direct_llm"
        assert len(reply1["content"]) > 10

        # Follow-up referencing previous answer
        reply2 = await service.send_message(
            conversation_id=conv["conversation_id"],
            content="Which of them has the most action items?",
        )
        assert reply2["llm_backend"] == "direct_llm"
        assert len(reply2["content"]) > 10

        # Verify history is persisted
        history = await service.get_conversation_history(conv["conversation_id"])
        assert len(history["messages"]) == 4  # 2 user + 2 assistant

        output = get_output_path("agent_multi_turn")
        write_output(
            output,
            f"Multi-turn conversation:\n"
            f"  Q1: Who attended this meeting?\n"
            f"  A1: {reply1['content'][:150]}...\n"
            f"  Q2: Which of them has the most action items?\n"
            f"  A2: {reply2['content'][:150]}...\n"
            f"  Total messages in DB: {len(history['messages'])}\n",
        )


# =============================================================================
# Test 3: Streaming with Real LLM
# =============================================================================


class TestStreamingRealLLM:
    """Test SSE streaming with real Ollama responses."""

    @pytest.mark.asyncio
    async def test_streaming_delivers_chunks(self, llm_client):
        """Test that chat_stream delivers real token chunks from Ollama."""
        chunks = []
        full_text = ""

        async for chunk in llm_client.chat_stream(
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line."}],
            max_tokens=64,
        ):
            chunks.append(chunk)
            if chunk.chunk:
                full_text += chunk.chunk

        # Should have multiple content chunks + final done chunk
        content_chunks = [c for c in chunks if c.chunk]
        done_chunks = [c for c in chunks if c.done]

        assert len(content_chunks) > 1, f"Expected multiple chunks, got {len(content_chunks)}"
        assert len(done_chunks) == 1
        assert done_chunks[0].processing_time_ms > 0
        assert len(full_text.strip()) > 0

        output = get_output_path("streaming_real_llm")
        write_output(
            output,
            f"Streaming from Ollama:\n"
            f"  total chunks: {len(chunks)}\n"
            f"  content chunks: {len(content_chunks)}\n"
            f"  full text: {full_text}\n"
            f"  processing_time_ms: {done_chunks[0].processing_time_ms:.1f}\n",
        )

    @pytest.mark.asyncio
    async def test_agent_streaming_via_service(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Test send_message_stream produces real SSE chunks from service layer."""
        service = intelligence_service_with_llm

        conv = await service.create_conversation(
            session_id=bot_session_id,
            title="Streaming Test",
            transcript_text=SAMPLE_TRANSCRIPT,
        )

        sse_chunks = []
        async for sse_line in service.send_message_stream(
            conversation_id=conv["conversation_id"],
            content="Summarize this meeting in 2 sentences.",
        ):
            sse_chunks.append(sse_line)

        assert len(sse_chunks) >= 2  # At least one content + final done

        # Parse the SSE data
        content_pieces = []
        final_chunk = None
        for line in sse_chunks:
            if line.startswith("data: "):
                data = json.loads(line[6:].strip())
                if data.get("chunk"):
                    content_pieces.append(data["chunk"])
                if data.get("done"):
                    final_chunk = data

        full_response = "".join(content_pieces)
        assert len(full_response) > 20, f"Response too short: {full_response}"
        assert final_chunk is not None
        assert final_chunk.get("message_id") is not None
        assert final_chunk.get("suggested_queries") is not None

        # Verify message was persisted in DB
        history = await service.get_conversation_history(conv["conversation_id"])
        assert len(history["messages"]) == 2  # user + assistant
        assert history["messages"][1]["content"] == full_response

        output = get_output_path("agent_streaming_service")
        write_output(
            output,
            f"Agent streaming via service:\n"
            f"  SSE lines: {len(sse_chunks)}\n"
            f"  content chunks: {len(content_pieces)}\n"
            f"  full response ({len(full_response)} chars): {full_response}\n"
            f"  message_id: {final_chunk.get('message_id')}\n"
            f"  persisted in DB: {len(history['messages'])} messages\n",
        )


# =============================================================================
# Test 4: Insight Generation with Real LLM
# =============================================================================


class TestInsightGenerationRealLLM:
    """Test post-meeting insight generation with real Ollama."""

    @pytest.mark.asyncio
    async def test_generate_meeting_summary(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Generate a real meeting summary insight from a transcript."""
        service = intelligence_service_with_llm
        await service.load_default_templates()

        insight = await service.generate_insight(
            session_id=bot_session_id,
            template_name="meeting_summary",
            transcript_text=SAMPLE_TRANSCRIPT,
            speakers=["Alice", "Bob", "Carol"],
            duration="15 minutes",
        )

        assert insight["content"] is not None
        assert len(insight["content"]) > 50
        assert insight["llm_backend"] == "direct_llm"
        assert insight["llm_model"] is not None
        assert insight["processing_time_ms"] > 0
        assert insight["insight_type"] == "summary"

        # Should reference meeting content
        content_lower = insight["content"].lower()
        has_meeting_content = any(
            keyword in content_lower
            for keyword in ["roadmap", "mobile", "api", "budget", "hire", "q3"]
        )
        assert (
            has_meeting_content
        ), f"Summary doesn't reference transcript. Got: {insight['content'][:200]}"

        output = get_output_path("insight_meeting_summary")
        write_output(
            output,
            f"Real meeting summary insight:\n"
            f"  insight_id: {insight['insight_id']}\n"
            f"  type: {insight['insight_type']}\n"
            f"  content ({len(insight['content'])} chars):\n{insight['content']}\n"
            f"  llm_backend: {insight['llm_backend']}\n"
            f"  processing_time_ms: {insight['processing_time_ms']:.1f}\n",
        )

    @pytest.mark.asyncio
    async def test_generate_action_items(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Generate real action items from a transcript."""
        service = intelligence_service_with_llm
        await service.load_default_templates()

        insight = await service.generate_insight(
            session_id=bot_session_id,
            template_name="action_items",
            transcript_text=SAMPLE_TRANSCRIPT,
            speakers=["Alice", "Bob", "Carol"],
        )

        assert len(insight["content"]) > 30
        assert insight["llm_backend"] == "direct_llm"

        # Should mention specific action items from the transcript
        content_lower = insight["content"].lower()
        has_actions = any(
            keyword in content_lower
            for keyword in ["api", "spec", "ticket", "hire", "spot", "review", "hr"]
        )
        assert (
            has_actions
        ), f"Action items don't reference transcript. Got: {insight['content'][:200]}"

        output = get_output_path("insight_action_items")
        write_output(
            output,
            f"Real action items insight:\n"
            f"  content ({len(insight['content'])} chars):\n{insight['content']}\n"
            f"  processing_time_ms: {insight['processing_time_ms']:.1f}\n",
        )

    @pytest.mark.asyncio
    async def test_generate_all_insights_parallel(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Generate multiple insights in parallel with real LLM."""
        service = intelligence_service_with_llm
        await service.load_default_templates()

        results = await service.generate_all_insights(
            session_id=bot_session_id,
            template_names=["meeting_summary", "action_items", "key_decisions"],
            transcript_text=SAMPLE_TRANSCRIPT,
            speakers=["Alice", "Bob", "Carol"],
            duration="15 minutes",
        )

        assert len(results) == 3
        successful = [r for r in results if "error" not in r]
        assert len(successful) == 3, f"Some insights failed: {results}"

        for r in successful:
            assert len(r["content"]) > 30
            assert r["llm_backend"] == "direct_llm"

        # Verify all persisted in DB
        all_insights = await service.get_insights(bot_session_id)
        assert len(all_insights) >= 3

        output = get_output_path("insight_parallel_generation")
        write_output(
            output,
            "Parallel insight generation (3 templates):\n"
            + "\n".join(
                f"  [{r.get('insight_type', 'unknown')}] "
                f"{r.get('title', 'N/A')}: {r['content'][:100]}..."
                for r in successful
            )
            + f"\n  Total persisted in DB: {len(all_insights)}\n",
        )


# =============================================================================
# Test 5: Auto-Note Generation with Real LLM
# =============================================================================


class TestAutoNoteRealLLM:
    """Test auto-note generation with real Ollama."""

    @pytest.mark.asyncio
    async def test_generate_auto_note_from_sentences(
        self, intelligence_service_with_llm, db_session_factory, bot_session_id
    ):
        """Generate a real auto-note from accumulated sentences."""
        service = intelligence_service_with_llm
        await service.load_default_templates()

        sentences = [
            {
                "text": "Let's discuss the Q3 roadmap.",
                "speaker_name": "Alice",
                "start_time": 0.0,
                "end_time": 3.5,
            },
            {
                "text": "I think we should prioritize the mobile app launch.",
                "speaker_name": "Bob",
                "start_time": 4.0,
                "end_time": 7.2,
            },
            {
                "text": "Engineering estimates 6 weeks for the MVP.",
                "speaker_name": "Carol",
                "start_time": 8.0,
                "end_time": 11.5,
            },
            {
                "text": "We need to finalize the API contracts by next Friday.",
                "speaker_name": "Carol",
                "start_time": 12.0,
                "end_time": 15.3,
            },
            {
                "text": "I can have the API spec ready by Wednesday.",
                "speaker_name": "Bob",
                "start_time": 16.0,
                "end_time": 19.0,
            },
        ]

        note = await service.generate_auto_note(
            session_id=bot_session_id,
            sentences=sentences,
        )

        assert note["note_type"] == "auto"
        assert len(note["content"]) > 20
        assert note["llm_backend"] == "direct_llm"
        assert note["llm_model"] is not None
        assert note["processing_time_ms"] > 0
        assert note["transcript_range_start"] == 0.0
        assert note["transcript_range_end"] == 19.0

        # Should summarize the discussion
        content_lower = note["content"].lower()
        has_relevant = any(
            keyword in content_lower for keyword in ["roadmap", "mobile", "api", "mvp", "week"]
        )
        assert has_relevant, f"Auto-note doesn't reference content. Got: {note['content'][:200]}"

        # Verify persisted
        notes = await service.get_notes(bot_session_id, note_type="auto")
        assert len(notes) == 1

        output = get_output_path("auto_note_real_llm")
        write_output(
            output,
            f"Real auto-note generation:\n"
            f"  note_id: {note['note_id']}\n"
            f"  note_type: {note['note_type']}\n"
            f"  content ({len(note['content'])} chars):\n{note['content']}\n"
            f"  time_range: {note['transcript_range_start']} - {note['transcript_range_end']}\n"
            f"  llm_backend: {note['llm_backend']}\n"
            f"  processing_time_ms: {note['processing_time_ms']:.1f}\n",
        )


# =============================================================================
# Test 6: translate_prompt Interface (for non-chat consumers)
# =============================================================================


class TestTranslatePromptRealLLM:
    """Test the translate_prompt Protocol interface with real Ollama."""

    @pytest.mark.asyncio
    async def test_translate_prompt_with_system_prompt(self, llm_client):
        """Test translate_prompt (the interface used by RollingWindowTranslator)."""
        result = await llm_client.translate_prompt(
            prompt="Translate to Spanish: The meeting starts at 3 PM.",
            system_prompt="You are a professional translator. Respond with only the translation.",
            max_tokens=64,
        )

        assert result.text is not None
        assert len(result.text.strip()) > 0
        assert result.processing_time_ms > 0

        output = get_output_path("translate_prompt_real")
        write_output(
            output,
            f"translate_prompt (Protocol interface):\n"
            f"  prompt: Translate to Spanish: The meeting starts at 3 PM.\n"
            f"  result: {result.text}\n"
            f"  processing_time_ms: {result.processing_time_ms:.1f}\n",
        )

    @pytest.mark.asyncio
    async def test_translate_prompt_stream_delivers_chunks(self, llm_client):
        """Test translate_prompt_stream delivers real streaming chunks."""
        chunks = []
        full_text = ""

        async for chunk in llm_client.translate_prompt_stream(
            prompt="List 3 colors, one per line.",
            max_tokens=32,
        ):
            chunks.append(chunk)
            if chunk.chunk:
                full_text += chunk.chunk

        content_chunks = [c for c in chunks if c.chunk]
        assert len(content_chunks) >= 1
        assert len(full_text.strip()) > 0

        output = get_output_path("translate_prompt_stream_real")
        write_output(
            output,
            f"translate_prompt_stream:\n"
            f"  chunks: {len(content_chunks)}\n"
            f"  full text: {full_text}\n",
        )


# =============================================================================
# Test 7: Full HTTP Endpoint with Real LLM
# =============================================================================


class TestHTTPEndpointRealLLM:
    """Test the FastAPI HTTP endpoints with real LLM behind them."""

    @pytest.fixture
    def test_app(self, intelligence_service_with_llm):
        from fastapi import FastAPI
        from routers.insights import get_intelligence_service, router

        app = FastAPI()
        app.include_router(router, prefix="/api/intelligence")
        app.dependency_overrides[get_intelligence_service] = lambda: intelligence_service_with_llm
        return app

    @pytest.fixture
    async def client(self, test_app):
        import httpx

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    @pytest.mark.asyncio
    async def test_full_http_agent_chat(self, client, bot_session_id):
        """Test complete HTTP flow: create conversation → send message → get real LLM answer."""
        # Create conversation with transcript
        conv_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={
                "title": "HTTP Real LLM Test",
                "transcript_text": SAMPLE_TRANSCRIPT,
            },
        )
        assert conv_resp.status_code == 201
        conv_id = conv_resp.json()["conversation_id"]

        # Send message and get real LLM response
        msg_resp = await client.post(
            f"/api/intelligence/agent/conversations/{conv_id}/messages",
            json={"content": "What decisions were made in this meeting?"},
        )
        assert msg_resp.status_code == 201
        data = msg_resp.json()

        assert data["role"] == "assistant"
        assert len(data["content"]) > 30
        assert data["llm_backend"] == "direct_llm"
        assert data["processing_time_ms"] > 0

        output = get_output_path("http_full_agent_real_llm")
        write_output(
            output,
            f"Full HTTP agent chat with real LLM:\n"
            f"  conversation_id: {conv_id}\n"
            f"  question: What decisions were made in this meeting?\n"
            f"  answer ({len(data['content'])} chars): {data['content']}\n"
            f"  llm_backend: {data['llm_backend']}\n"
            f"  processing_time_ms: {data['processing_time_ms']:.1f}\n",
        )

    @pytest.mark.asyncio
    async def test_http_streaming_real_llm(self, client, bot_session_id):
        """Test SSE streaming endpoint with real LLM tokens."""
        # Create conversation
        conv_resp = await client.post(
            f"/api/intelligence/sessions/{bot_session_id}/agent/conversations",
            json={"title": "HTTP Stream Test", "transcript_text": SAMPLE_TRANSCRIPT},
        )
        conv_id = conv_resp.json()["conversation_id"]

        # Stream response
        stream_resp = await client.post(
            f"/api/intelligence/agent/conversations/{conv_id}/messages/stream",
            json={"content": "Who is responsible for the API spec?"},
        )
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")

        # Parse SSE
        body = stream_resp.text
        data_lines = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(data_lines) >= 2  # At least some content + done

        content_pieces = []
        final = None
        for line in data_lines:
            parsed = json.loads(line[6:])
            if parsed.get("chunk"):
                content_pieces.append(parsed["chunk"])
            if parsed.get("done"):
                final = parsed

        full_response = "".join(content_pieces)
        assert len(full_response) > 10
        assert final is not None
        assert final.get("message_id") is not None

        output = get_output_path("http_streaming_real_llm")
        write_output(
            output,
            f"HTTP SSE streaming with real LLM:\n"
            f"  SSE lines: {len(data_lines)}\n"
            f"  content chunks: {len(content_pieces)}\n"
            f"  full response: {full_response}\n"
            f"  message_id: {final.get('message_id')}\n",
        )
