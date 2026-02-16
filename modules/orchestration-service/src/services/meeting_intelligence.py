"""
Meeting Intelligence Service

Central service for meeting analysis features:
- Auto-generated notes during meetings (LLM-powered)
- Manual annotations and LLM-analyzed notes
- Post-meeting insight generation from configurable templates
- Agent conversation with real LLM integration for Q&A about transcripts

LLM calls go through the unified LLMClient, which supports:
- Direct mode: OpenAI-compatible endpoint (preferred for agent chat)
- Proxy mode: Translation Service V3 API
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from string import Template
from typing import Any

import yaml
from clients.llm_client import LLMClient
from clients.protocol import LLMClientProtocol
from config import MeetingIntelligenceSettings
from database.models import (
    AgentConversation,
    AgentMessage,
    InsightPromptTemplate,
    MeetingInsight,
    MeetingNote,
)
from livetranslate_common.logging import get_logger
from sqlalchemy import select

logger = get_logger()


class MeetingIntelligenceService:
    """
    Central service for all meeting intelligence features.

    Uses LLMClientProtocol as a generic LLM gateway for all analysis calls.
    Persists all results to PostgreSQL via async SQLAlchemy sessions.
    """

    def __init__(
        self,
        db_session_factory,
        translation_client: LLMClientProtocol | None = None,
        settings: MeetingIntelligenceSettings | None = None,
    ):
        self.db_session_factory = db_session_factory
        self.translation_client = translation_client
        self.settings = settings or MeetingIntelligenceSettings()
        self._templates_loaded = False

    # =========================================================================
    # Template Management
    # =========================================================================

    async def load_default_templates(self) -> int:
        """
        Load default templates from YAML config and seed into DB.

        Only inserts templates that don't already exist (by name).
        Returns the number of templates seeded.
        """
        templates_path = Path(self.settings.templates_config_path)
        if not templates_path.is_absolute():
            # Resolve relative to orchestration-service root
            base = Path(__file__).parent.parent.parent
            templates_path = base / templates_path

        if not templates_path.exists():
            logger.warning(f"Templates config not found: {templates_path}")
            return 0

        with open(templates_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        templates = data.get("templates", [])
        seeded = 0

        async with self.db_session_factory() as session:
            for tmpl_data in templates:
                name = tmpl_data["name"]
                # Check if already exists
                result = await session.execute(
                    select(InsightPromptTemplate).where(InsightPromptTemplate.name == name)
                )
                existing = result.scalar_one_or_none()
                if existing:
                    continue

                template = InsightPromptTemplate(
                    template_id=uuid.uuid4(),
                    name=name,
                    description=tmpl_data.get("description", ""),
                    category=tmpl_data.get("category", "custom"),
                    prompt_template=tmpl_data["prompt_template"],
                    system_prompt=tmpl_data.get("system_prompt"),
                    expected_output_format=tmpl_data.get("expected_output_format", "markdown"),
                    default_temperature=tmpl_data.get("default_temperature", 0.3),
                    default_max_tokens=tmpl_data.get("default_max_tokens", 1024),
                    is_builtin=True,
                    is_active=True,
                )
                session.add(template)
                seeded += 1

            await session.commit()

        if seeded:
            logger.info(f"Seeded {seeded} default insight templates")
        self._templates_loaded = True
        return seeded

    async def get_templates(
        self, category: str | None = None, active_only: bool = True
    ) -> list[dict[str, Any]]:
        """Get all templates, optionally filtered."""
        async with self.db_session_factory() as session:
            query = select(InsightPromptTemplate)
            if category:
                query = query.where(InsightPromptTemplate.category == category)
            if active_only:
                query = query.where(InsightPromptTemplate.is_active.is_(True))
            query = query.order_by(InsightPromptTemplate.category, InsightPromptTemplate.name)

            result = await session.execute(query)
            return [t.to_dict() for t in result.scalars().all()]

    async def get_template(self, template_id_or_name: str) -> dict[str, Any] | None:
        """Get a template by ID or name."""
        async with self.db_session_factory() as session:
            # Try by name first
            result = await session.execute(
                select(InsightPromptTemplate).where(
                    InsightPromptTemplate.name == template_id_or_name
                )
            )
            template = result.scalar_one_or_none()

            if not template:
                # Try by UUID
                try:
                    tid = uuid.UUID(template_id_or_name)
                    result = await session.execute(
                        select(InsightPromptTemplate).where(
                            InsightPromptTemplate.template_id == tid
                        )
                    )
                    template = result.scalar_one_or_none()
                except ValueError:
                    pass

            return template.to_dict() if template else None

    async def create_template(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a custom template."""
        # Validate that template contains $transcript variable
        prompt_template = data.get("prompt_template", "")
        if "$transcript" not in prompt_template:
            raise ValueError("Template prompt_template must contain the $transcript variable")

        async with self.db_session_factory() as session:
            template = InsightPromptTemplate(
                template_id=uuid.uuid4(),
                name=data["name"],
                description=data.get("description"),
                category=data.get("category", "custom"),
                prompt_template=data["prompt_template"],
                system_prompt=data.get("system_prompt"),
                expected_output_format=data.get("expected_output_format", "markdown"),
                default_llm_backend=data.get("default_llm_backend"),
                default_temperature=data.get("default_temperature", 0.3),
                default_max_tokens=data.get("default_max_tokens", 1024),
                is_builtin=False,
                is_active=True,
                template_metadata=data.get("metadata"),
            )
            session.add(template)
            await session.commit()
            await session.refresh(template)
            return template.to_dict()

    async def update_template(
        self, template_id: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update a template."""
        async with self.db_session_factory() as session:
            try:
                tid = uuid.UUID(template_id)
            except ValueError:
                return None

            result = await session.execute(
                select(InsightPromptTemplate).where(InsightPromptTemplate.template_id == tid)
            )
            template = result.scalar_one_or_none()
            if not template:
                return None

            for key, value in data.items():
                if value is not None and hasattr(template, key):
                    setattr(template, key, value)
                elif key == "metadata" and value is not None:
                    template.template_metadata = value

            await session.commit()
            await session.refresh(template)
            return template.to_dict()

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template (only non-builtin)."""
        async with self.db_session_factory() as session:
            try:
                tid = uuid.UUID(template_id)
            except ValueError:
                return False

            result = await session.execute(
                select(InsightPromptTemplate).where(InsightPromptTemplate.template_id == tid)
            )
            template = result.scalar_one_or_none()
            if not template or template.is_builtin:
                return False

            await session.delete(template)
            await session.commit()
            return True

    # =========================================================================
    # Notes (Real-Time)
    # =========================================================================

    async def create_manual_note(
        self,
        session_id: str,
        content: str,
        speaker_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a plain-text manual note."""
        async with self.db_session_factory() as session:
            note = MeetingNote(
                note_id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                note_type="manual",
                content=content,
                speaker_name=speaker_name,
            )
            session.add(note)
            await session.commit()
            await session.refresh(note)
            return note.to_dict()

    async def create_analyzed_note(
        self,
        session_id: str,
        prompt: str,
        context_sentences: list[str] | None = None,
        speaker_name: str | None = None,
        llm_backend: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Create an LLM-analyzed note from a user prompt."""
        if not self.translation_client:
            raise RuntimeError("Translation client not configured for LLM calls")

        backend = llm_backend or self.settings.default_llm_backend
        temp = temperature if temperature is not None else self.settings.default_temperature
        tokens = max_tokens or self.settings.default_max_tokens

        # Build context into the prompt
        full_prompt = prompt
        if context_sentences:
            context_text = "\n".join(f"- {s}" for s in context_sentences)
            full_prompt = f"{prompt}\n\nContext from transcript:\n{context_text}"

        start_ms = time.monotonic()
        result = await self.translation_client.translate_prompt(
            prompt=full_prompt,
            backend=backend,
            temperature=temp,
            max_tokens=tokens,
        )
        elapsed_ms = (time.monotonic() - start_ms) * 1000

        async with self.db_session_factory() as session:
            note = MeetingNote(
                note_id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                note_type="annotation",
                content=result.text,
                prompt_used=full_prompt,
                context_sentences=context_sentences,
                speaker_name=speaker_name,
                llm_backend=result.backend_used,
                llm_model=result.model_used,
                processing_time_ms=elapsed_ms,
            )
            session.add(note)
            await session.commit()
            await session.refresh(note)
            return note.to_dict()

    async def generate_auto_note(
        self,
        session_id: str,
        sentences: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate an auto-note from accumulated sentences.

        Called by the pipeline coordinator every N sentences.
        """
        if not self.translation_client:
            raise RuntimeError("Translation client not configured for LLM calls")

        # Load template
        template_data = await self.get_template(self.settings.auto_notes_template)
        if not template_data:
            raise ValueError(f"Auto-note template '{self.settings.auto_notes_template}' not found")

        # Build transcript from sentences
        transcript_lines = []
        speakers = set()
        for s in sentences:
            speaker = s.get("speaker_name", "Unknown")
            speakers.add(speaker)
            transcript_lines.append(f"{speaker}: {s.get('text', '')}")

        transcript_text = "\n".join(transcript_lines)
        speakers_str = ", ".join(sorted(speakers))

        # Substitute template variables (safe_substitute prevents KeyError from user content)
        prompt = Template(template_data["prompt_template"]).safe_substitute(
            transcript=transcript_text,
            speakers=speakers_str,
            duration="ongoing",
            language="auto-detected",
            custom_instructions="",
        )
        system_prompt = template_data.get("system_prompt")

        backend = self.settings.default_llm_backend
        temp = template_data.get("default_temperature", 0.3)
        tokens = template_data.get("default_max_tokens", 256)

        start_ms = time.monotonic()
        result = await self.translation_client.translate_prompt(
            prompt=prompt,
            backend=backend,
            temperature=temp,
            max_tokens=tokens,
            system_prompt=system_prompt,
        )
        elapsed_ms = (time.monotonic() - start_ms) * 1000

        # Compute time range
        start_times = [s.get("start_time", 0.0) for s in sentences]
        end_times = [s.get("end_time", 0.0) for s in sentences]

        context_texts = [s.get("text", "") for s in sentences]

        async with self.db_session_factory() as session:
            note = MeetingNote(
                note_id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                note_type="auto",
                content=result.text,
                prompt_used=prompt,
                context_sentences=context_texts,
                transcript_range_start=min(start_times) if start_times else None,
                transcript_range_end=max(end_times) if end_times else None,
                llm_backend=result.backend_used,
                llm_model=result.model_used,
                processing_time_ms=elapsed_ms,
            )
            session.add(note)
            await session.commit()
            await session.refresh(note)
            return note.to_dict()

    async def get_notes(
        self,
        session_id: str,
        note_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get notes for a session."""
        async with self.db_session_factory() as session:
            query = select(MeetingNote).where(MeetingNote.session_id == uuid.UUID(session_id))
            if note_type:
                query = query.where(MeetingNote.note_type == note_type)
            query = query.order_by(MeetingNote.created_at.desc())

            result = await session.execute(query)
            return [n.to_dict() for n in result.scalars().all()]

    async def delete_note(self, note_id: str) -> bool:
        """Delete a note."""
        async with self.db_session_factory() as session:
            try:
                nid = uuid.UUID(note_id)
            except ValueError:
                return False

            result = await session.execute(select(MeetingNote).where(MeetingNote.note_id == nid))
            note = result.scalar_one_or_none()
            if not note:
                return False

            await session.delete(note)
            await session.commit()
            return True

    # =========================================================================
    # Post-Meeting Insights
    # =========================================================================

    async def generate_insight(
        self,
        session_id: str,
        template_name: str,
        transcript_text: str,
        speakers: list[str] | None = None,
        duration: str | None = None,
        language: str = "en",
        custom_instructions: str | None = None,
        llm_backend: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate a single insight using a template.

        Args:
            session_id: Session to generate insight for
            template_name: Name of the template to use
            transcript_text: Full transcript text
            speakers: List of speaker names
            duration: Meeting duration string
            language: Primary language
            custom_instructions: Additional instructions
            llm_backend: LLM backend override
            temperature: Temperature override
            max_tokens: Max tokens override
        """
        if not self.translation_client:
            raise RuntimeError("Translation client not configured for LLM calls")

        # Load template
        template_data = await self.get_template(template_name)
        if not template_data:
            raise ValueError(f"Template '{template_name}' not found")

        # Truncate transcript if needed
        max_chars = self.settings.max_transcript_chars_for_insight
        was_truncated = len(transcript_text) > max_chars
        truncated = transcript_text[:max_chars] if was_truncated else transcript_text
        if was_truncated:
            logger.warning(
                f"Transcript truncated from {len(transcript_text)} to {max_chars} chars "
                f"for template '{template_name}'"
            )

        # Substitute template variables (safe_substitute prevents KeyError from user content)
        speakers_str = ", ".join(speakers) if speakers else "Unknown"
        prompt = Template(template_data["prompt_template"]).safe_substitute(
            transcript=truncated,
            speakers=speakers_str,
            duration=duration or "Unknown",
            language=language,
            custom_instructions=custom_instructions or "",
        )
        system_prompt = template_data.get("system_prompt")

        backend = (
            llm_backend
            or template_data.get("default_llm_backend")
            or self.settings.insights_llm_backend
            or self.settings.default_llm_backend
        )
        temp = (
            temperature
            if temperature is not None
            else template_data.get("default_temperature", 0.3)
        )
        tokens = max_tokens or template_data.get("default_max_tokens", 1024)

        start_ms = time.monotonic()
        result = await self.translation_client.translate_prompt(
            prompt=prompt,
            backend=backend,
            temperature=temp,
            max_tokens=tokens,
            system_prompt=system_prompt,
        )
        elapsed_ms = (time.monotonic() - start_ms) * 1000

        # Parse template_id
        template_id = None
        try:
            template_id = uuid.UUID(template_data["template_id"])
        except (ValueError, KeyError):
            pass

        # Build title from template name
        title = template_data.get("description", template_name.replace("_", " ").title())

        async with self.db_session_factory() as session:
            insight = MeetingInsight(
                insight_id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                template_id=template_id,
                insight_type=template_data.get("category", "custom"),
                title=title,
                content=result.text,
                prompt_used=prompt,
                transcript_length=len(truncated),
                llm_backend=result.backend_used,
                llm_model=result.model_used,
                processing_time_ms=elapsed_ms,
                insight_metadata={
                    "speakers": speakers,
                    "duration": duration,
                    "language": language,
                    "template_name": template_name,
                    "transcript_truncated": was_truncated,
                    "transcript_original_length": len(transcript_text),
                },
            )
            session.add(insight)
            await session.commit()
            await session.refresh(insight)
            return insight.to_dict()

    async def generate_all_insights(
        self,
        session_id: str,
        template_names: list[str],
        transcript_text: str,
        speakers: list[str] | None = None,
        duration: str | None = None,
        language: str = "en",
        custom_instructions: str | None = None,
        llm_backend: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate insights from multiple templates in parallel."""
        # Limit concurrency to avoid flooding LLM backend
        semaphore = asyncio.Semaphore(3)

        async def _generate_one(name: str) -> dict[str, Any]:
            async with semaphore:
                try:
                    return await self.generate_insight(
                        session_id=session_id,
                        template_name=name,
                        transcript_text=transcript_text,
                        speakers=speakers,
                        duration=duration,
                        language=language,
                        custom_instructions=custom_instructions,
                        llm_backend=llm_backend,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    logger.error(f"Failed to generate insight from template '{name}': {e}")
                    return {
                        "error": str(e),
                        "template_name": name,
                    }

        results = await asyncio.gather(*[_generate_one(name) for name in template_names])
        return list(results)

    async def get_insights(self, session_id: str) -> list[dict[str, Any]]:
        """Get all insights for a session."""
        async with self.db_session_factory() as session:
            result = await session.execute(
                select(MeetingInsight)
                .where(MeetingInsight.session_id == uuid.UUID(session_id))
                .order_by(MeetingInsight.created_at.desc())
            )
            return [i.to_dict() for i in result.scalars().all()]

    async def get_insight(self, insight_id: str) -> dict[str, Any] | None:
        """Get a specific insight."""
        async with self.db_session_factory() as session:
            try:
                iid = uuid.UUID(insight_id)
            except ValueError:
                return None

            result = await session.execute(
                select(MeetingInsight).where(MeetingInsight.insight_id == iid)
            )
            insight = result.scalar_one_or_none()
            return insight.to_dict() if insight else None

    async def delete_insight(self, insight_id: str) -> bool:
        """Delete an insight."""
        async with self.db_session_factory() as session:
            try:
                iid = uuid.UUID(insight_id)
            except ValueError:
                return False

            result = await session.execute(
                select(MeetingInsight).where(MeetingInsight.insight_id == iid)
            )
            insight = result.scalar_one_or_none()
            if not insight:
                return False

            await session.delete(insight)
            await session.commit()
            return True

    # =========================================================================
    # Agent Conversation Scaffolding
    # =========================================================================

    async def create_conversation(
        self,
        session_id: str,
        title: str | None = None,
        transcript_text: str | None = None,
    ) -> dict[str, Any]:
        """Create a new agent conversation for a meeting session."""
        system_context = None
        if transcript_text:
            # Truncate for system context
            max_ctx = self.settings.max_transcript_chars_for_insight
            truncated = (
                transcript_text[:max_ctx] if len(transcript_text) > max_ctx else transcript_text
            )
            system_context = (
                f"You are an AI assistant helping analyze a meeting transcript. "
                f"Answer questions based on the following transcript:\n\n{truncated}"
            )

        async with self.db_session_factory() as session:
            conversation = AgentConversation(
                conversation_id=uuid.uuid4(),
                session_id=uuid.UUID(session_id),
                title=title or "Meeting Q&A",
                status="active",
                system_context=system_context,
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation.to_dict()

    async def send_message(
        self,
        conversation_id: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Send a message in an agent conversation and get an LLM response.

        Supports both direct mode (native multi-turn chat) and
        proxy mode (flattened prompt fallback) via the unified LLMClient.
        Falls back to a helpful error message if no LLM client is available.
        """
        cid = uuid.UUID(conversation_id)

        async with self.db_session_factory() as session:
            # Fetch conversation for system context
            result = await session.execute(
                select(AgentConversation).where(AgentConversation.conversation_id == cid)
            )
            conversation = result.scalar_one_or_none()
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            # Fetch message history
            msg_result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == cid)
                .order_by(AgentMessage.created_at.asc())
            )
            history = msg_result.scalars().all()

            # Store user message
            user_msg = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="user",
                content=content,
            )
            session.add(user_msg)
            await session.flush()

            # Build messages array for LLM
            messages = _build_conversation_messages(
                system_context=conversation.system_context,
                message_history=list(history),
                new_user_message=content,
                max_tokens=self.settings.agent_max_context_tokens,
            )

            # Call LLM
            response_text = ""
            llm_backend = None
            llm_model = None
            processing_time_ms = None

            if self.translation_client:
                start_ms = time.monotonic()
                try:
                    if (
                        isinstance(self.translation_client, LLMClient)
                        and not self.translation_client.proxy_mode
                    ):
                        # LLMClient in direct mode - native multi-turn
                        result = await self.translation_client.chat(
                            messages=messages,
                            max_tokens=self.settings.default_max_tokens,
                            temperature=self.settings.default_temperature,
                        )
                    else:
                        # LLMClient in proxy mode or generic LLMClientProtocol - flatten to prompt
                        prompt, system_prompt = _flatten_messages_to_prompt(messages)
                        result = await self.translation_client.translate_prompt(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=self.settings.default_max_tokens,
                            temperature=self.settings.default_temperature,
                        )

                    response_text = result.text
                    llm_backend = result.backend_used
                    llm_model = result.model_used
                    processing_time_ms = (time.monotonic() - start_ms) * 1000

                except Exception as e:
                    logger.error(f"LLM call failed for conversation {conversation_id}: {e}")
                    response_text = (
                        f"I'm sorry, I couldn't process your question due to an error: {e!s}. "
                        "Please try again or check that the LLM backend is running."
                    )
                    processing_time_ms = (time.monotonic() - start_ms) * 1000
            else:
                response_text = (
                    "No LLM backend is configured. To enable AI responses, "
                    "set INTELLIGENCE_DIRECT_LLM_ENABLED=true with a running Ollama/OpenAI backend, "
                    "or ensure the Translation Service is running on port 5003."
                )

            # Generate follow-up suggestions
            suggested = _generate_suggested_queries(
                response_text, conversation.title or "Meeting Q&A"
            )

            # Store assistant message
            assistant_msg = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="assistant",
                content=response_text,
                llm_backend=llm_backend,
                llm_model=llm_model,
                processing_time_ms=processing_time_ms,
                suggested_queries=suggested,
                message_metadata={
                    "context_messages": len(messages),
                    "has_llm": self.translation_client is not None,
                },
            )
            session.add(assistant_msg)
            await session.commit()
            await session.refresh(assistant_msg)

            return {
                "message_id": str(assistant_msg.message_id),
                "conversation_id": str(assistant_msg.conversation_id),
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "llm_backend": assistant_msg.llm_backend,
                "llm_model": assistant_msg.llm_model,
                "processing_time_ms": assistant_msg.processing_time_ms,
                "suggested_queries": assistant_msg.suggested_queries,
                "created_at": assistant_msg.created_at.isoformat()
                if assistant_msg.created_at
                else None,
            }

    async def send_message_stream(
        self,
        conversation_id: str,
        content: str,
    ) -> AsyncIterator[str]:
        """
        Stream an agent response via Server-Sent Events.

        Yields SSE-formatted strings for progressive response delivery.
        """
        cid = uuid.UUID(conversation_id)

        async with self.db_session_factory() as session:
            # Fetch conversation
            result = await session.execute(
                select(AgentConversation).where(AgentConversation.conversation_id == cid)
            )
            conversation = result.scalar_one_or_none()
            if not conversation:
                yield f"data: {json.dumps({'error': 'Conversation not found', 'done': True})}\n\n"
                return

            # Fetch message history
            msg_result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == cid)
                .order_by(AgentMessage.created_at.asc())
            )
            history = msg_result.scalars().all()

            # Store user message immediately
            user_msg = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="user",
                content=content,
            )
            session.add(user_msg)
            await session.flush()

            # Build messages for LLM
            messages = _build_conversation_messages(
                system_context=conversation.system_context,
                message_history=list(history),
                new_user_message=content,
                max_tokens=self.settings.agent_max_context_tokens,
            )

            # Stream from LLM
            accumulated_text = ""
            llm_backend = None
            llm_model = None
            processing_time_ms = None

            if self.translation_client:
                try:
                    if (
                        isinstance(self.translation_client, LLMClient)
                        and not self.translation_client.proxy_mode
                    ):
                        stream = self.translation_client.chat_stream(
                            messages=messages,
                            max_tokens=self.settings.default_max_tokens,
                            temperature=self.settings.default_temperature,
                        )
                    else:
                        prompt, system_prompt = _flatten_messages_to_prompt(messages)
                        stream = self.translation_client.translate_prompt_stream(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=self.settings.default_max_tokens,
                            temperature=self.settings.default_temperature,
                        )

                    async for chunk in stream:
                        if chunk.error:
                            yield f"data: {json.dumps({'error': chunk.error, 'done': True})}\n\n"
                            return

                        if chunk.chunk:
                            accumulated_text += chunk.chunk
                            yield f"data: {json.dumps({'chunk': chunk.chunk, 'done': False})}\n\n"

                        if chunk.done:
                            llm_backend = chunk.backend_used
                            llm_model = chunk.model_used
                            processing_time_ms = chunk.processing_time_ms
                            break

                except Exception as e:
                    logger.error(f"Streaming LLM call failed: {e}")
                    error_text = f"LLM error: {e!s}"
                    yield f"data: {json.dumps({'error': error_text, 'done': True})}\n\n"
                    accumulated_text = error_text
            else:
                no_llm_msg = (
                    "No LLM backend configured. Set INTELLIGENCE_DIRECT_LLM_ENABLED=true "
                    "or ensure Translation Service is running."
                )
                yield f"data: {json.dumps({'chunk': no_llm_msg, 'done': False})}\n\n"
                accumulated_text = no_llm_msg

            if not accumulated_text:
                accumulated_text = "No response generated."

            # Generate suggestions
            suggested = _generate_suggested_queries(
                accumulated_text, conversation.title or "Meeting Q&A"
            )

            # Store assistant message
            assistant_msg = AgentMessage(
                message_id=uuid.uuid4(),
                conversation_id=cid,
                role="assistant",
                content=accumulated_text,
                llm_backend=llm_backend,
                llm_model=llm_model,
                processing_time_ms=processing_time_ms,
                suggested_queries=suggested,
                message_metadata={
                    "streamed": True,
                    "context_messages": len(messages),
                },
            )
            session.add(assistant_msg)
            await session.commit()
            await session.refresh(assistant_msg)

            # Final SSE with message metadata
            yield f"data: {json.dumps({'done': True, 'message_id': str(assistant_msg.message_id), 'suggested_queries': suggested})}\n\n"

    async def get_conversation_history(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation with full message history."""
        cid = uuid.UUID(conversation_id)

        async with self.db_session_factory() as session:
            result = await session.execute(
                select(AgentConversation).where(AgentConversation.conversation_id == cid)
            )
            conversation = result.scalar_one_or_none()
            if not conversation:
                return None

            msg_result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == cid)
                .order_by(AgentMessage.created_at.asc())
            )
            messages = msg_result.scalars().all()

            conv_dict = conversation.to_dict()
            conv_dict["messages"] = [
                {
                    "message_id": str(m.message_id),
                    "conversation_id": str(m.conversation_id),
                    "role": m.role,
                    "content": m.content,
                    "llm_backend": m.llm_backend,
                    "llm_model": m.llm_model,
                    "processing_time_ms": m.processing_time_ms,
                    "suggested_queries": m.suggested_queries,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ]
            return conv_dict

    async def get_suggested_queries(self, session_id: str) -> list[str]:
        """Return pre-defined suggested queries for a session."""
        return [
            "Summarize this meeting",
            "What are the action items?",
            "What decisions were made?",
            "What questions remain unanswered?",
            "Analyze each speaker's contributions",
            "What are the key takeaways?",
        ]


# =============================================================================
# Module-Level Helper Functions
# =============================================================================


def _build_conversation_messages(
    system_context: str | None,
    message_history: list,
    new_user_message: str,
    max_tokens: int = 8192,
) -> list[dict[str, str]]:
    """
    Build a messages array for LLM consumption with context window management.

    Always includes:
    1. System context (if present)
    2. The new user message (latest)

    Fills remaining budget with history messages (newest first),
    using approximate token estimation (len(text) / 4).

    Args:
        system_context: System prompt with transcript context
        message_history: List of AgentMessage ORM objects (ordered by created_at asc)
        new_user_message: The new user message to include
        max_tokens: Maximum context window in approximate tokens

    Returns:
        List of {"role": "...", "content": "..."} dicts
    """
    messages: list[dict[str, str]] = []
    budget = max_tokens

    # System context always included
    if system_context:
        system_tokens = len(system_context) // 4
        messages.append({"role": "system", "content": system_context})
        budget -= system_tokens

    # New user message always included
    user_tokens = len(new_user_message) // 4
    budget -= user_tokens

    # Fill remaining budget with history (newest first for relevance)
    history_messages: list[dict[str, str]] = []
    for msg in reversed(message_history):
        msg_text = msg.content if hasattr(msg, "content") else str(msg)
        msg_role = msg.role if hasattr(msg, "role") else "user"
        msg_tokens = len(msg_text) // 4

        if budget - msg_tokens < 0:
            break

        history_messages.insert(0, {"role": msg_role, "content": msg_text})
        budget -= msg_tokens

    messages.extend(history_messages)
    messages.append({"role": "user", "content": new_user_message})

    return messages


def _flatten_messages_to_prompt(
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """
    Flatten a messages array into (prompt, system_prompt) for proxy mode.

    The system message becomes the system_prompt parameter.
    All other messages are concatenated as "ROLE: content" lines.

    Returns:
        Tuple of (prompt_string, system_prompt_string)
    """
    system_prompt = ""
    conversation_lines = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_prompt = content
        elif role == "user":
            conversation_lines.append(f"USER: {content}")
        elif role == "assistant":
            conversation_lines.append(f"ASSISTANT: {content}")

    prompt = "\n".join(conversation_lines)
    return prompt, system_prompt


def _generate_suggested_queries(
    response_text: str,
    conversation_title: str,
) -> list[str]:
    """
    Generate contextual follow-up suggestions based on the conversation.

    Uses heuristic keyword matching to provide relevant follow-ups.
    """
    suggestions = []
    response_lower = response_text.lower()

    # Always include some base suggestions
    base = [
        "Can you elaborate on that?",
        "What are the next steps?",
    ]

    # Topic-aware suggestions
    if any(w in response_lower for w in ["action", "task", "todo", "assign"]):
        suggestions.append("Who is responsible for each action item?")
        suggestions.append("What are the deadlines?")
    if any(w in response_lower for w in ["decision", "agreed", "decided", "consensus"]):
        suggestions.append("What were the alternatives considered?")
        suggestions.append("Are there any risks with these decisions?")
    if any(w in response_lower for w in ["risk", "concern", "issue", "problem"]):
        suggestions.append("How can we mitigate these risks?")
        suggestions.append("What is the priority order?")
    if any(w in response_lower for w in ["budget", "cost", "spend", "invest"]):
        suggestions.append("What is the total budget impact?")
    if any(w in response_lower for w in ["timeline", "schedule", "deadline", "date"]):
        suggestions.append("Are there any scheduling conflicts?")
    if any(w in response_lower for w in ["speaker", "said", "mentioned", "participant"]):
        suggestions.append("What did each speaker contribute?")

    # Combine and limit
    all_suggestions = suggestions + base
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in all_suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:5]
