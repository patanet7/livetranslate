"""
Voice Command Interceptor for Fireflies Real-Time Pipeline.

Detects spoken voice commands in the transcript stream and executes them.
Config-driven: only active when PipelineConfig.voice_commands_enabled=True.

Supported commands (case-insensitive, prefix-matched):
    "{prefix} pause"              → Pause pipeline processing
    "{prefix} resume"             → Resume pipeline processing
    "{prefix} language {lang}"    → Switch target language
    "{prefix} display {mode}"     → Change display mode (english/translated/both)

Where {prefix} defaults to "LiveTranslate" (configurable via voice_command_prefix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from livetranslate_common.logging import get_logger

if TYPE_CHECKING:
    from src.services.pipeline.config import PipelineConfig
    from src.services.pipeline.coordinator import TranscriptionPipelineCoordinator

logger = get_logger()

# Language name → ISO code mapping for voice commands
LANGUAGE_ALIASES: dict[str, str] = {
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "russian": "ru",
    "hindi": "hi",
    "dutch": "nl",
    "swedish": "sv",
    "polish": "pl",
    "turkish": "tr",
    "english": "en",
}


class CommandInterceptor:
    """Detects and executes voice commands in the transcript stream.

    Usage:
        interceptor = CommandInterceptor(config=pipeline_config, coordinator=coordinator)

        # In the chunk processing path:
        if interceptor.check(chunk_text):
            result = await interceptor.execute(chunk_text)
            # Don't pass this chunk to the pipeline
    """

    def __init__(
        self,
        config: PipelineConfig,
        coordinator: TranscriptionPipelineCoordinator,
        ws_broadcast: Any | None = None,
        session_id: str = "",
    ):
        self._config = config
        self._coordinator = coordinator
        self._ws_broadcast = ws_broadcast
        self._session_id = session_id or config.session_id
        self._commands_executed: int = 0

    @property
    def enabled(self) -> bool:
        return self._config.voice_commands_enabled

    @property
    def prefix(self) -> str:
        return self._config.voice_command_prefix.lower().strip()

    @property
    def commands_executed(self) -> int:
        return self._commands_executed

    def check(self, text: str) -> bool:
        """Check if text starts with the voice command prefix.

        Returns True if this text is a voice command (should be intercepted).
        Returns False if it's normal speech (pass through to pipeline).
        """
        if not self.enabled:
            return False
        return text.strip().lower().startswith(self.prefix)

    async def execute(self, text: str) -> dict[str, Any]:
        """Parse and execute a voice command.

        Returns a dict describing the action taken.
        Caller should NOT pass this chunk to the pipeline.
        """
        cleaned = text.strip()
        # Strip prefix (case-insensitive)
        after_prefix = cleaned[len(self.prefix) :].strip().lower()

        # Remove trailing punctuation from ASR
        after_prefix = after_prefix.rstrip(".,!?;:")

        result: dict[str, Any] = {"command": after_prefix, "executed": False}

        if after_prefix in ("pause", "stop"):
            self._coordinator.pause()
            result = {"command": "pause", "executed": True}
            logger.info("voice_command_executed", command="pause", session_id=self._session_id)

        elif after_prefix in ("resume", "start", "go", "continue"):
            self._coordinator.resume()
            result = {"command": "resume", "executed": True}
            logger.info("voice_command_executed", command="resume", session_id=self._session_id)

        elif after_prefix.startswith("language ") or after_prefix.startswith("lang "):
            lang_arg = after_prefix.split(maxsplit=1)[1].strip()
            lang_code = LANGUAGE_ALIASES.get(lang_arg, lang_arg)
            self._coordinator.config.target_languages = [lang_code]
            result = {"command": "language", "language": lang_code, "executed": True}
            logger.info(
                "voice_command_executed",
                command="language",
                language=lang_code,
                session_id=self._session_id,
            )

        elif after_prefix.startswith("display ") or after_prefix.startswith("mode "):
            mode_arg = after_prefix.split(maxsplit=1)[1].strip()
            if mode_arg in ("english", "translated", "both"):
                self._coordinator.config.display_mode = mode_arg
                result = {"command": "display", "mode": mode_arg, "executed": True}
                logger.info(
                    "voice_command_executed",
                    command="display",
                    mode=mode_arg,
                    session_id=self._session_id,
                )
            else:
                result = {"command": "display", "error": f"unknown mode: {mode_arg}", "executed": False}
                logger.warning(
                    "voice_command_unknown_mode",
                    mode=mode_arg,
                    session_id=self._session_id,
                )

        else:
            result = {"command": after_prefix, "error": "unrecognized", "executed": False}
            logger.warning(
                "voice_command_unrecognized",
                command=after_prefix,
                session_id=self._session_id,
            )

        if result.get("executed"):
            self._commands_executed += 1

            # Broadcast command event to WebSocket clients
            if self._ws_broadcast:
                try:
                    await self._ws_broadcast(
                        self._session_id,
                        {
                            "event": "voice_command",
                            "command": result.get("command"),
                            "details": result,
                        },
                    )
                except Exception as e:
                    logger.warning("voice_command_broadcast_failed", error=str(e))

        return result
