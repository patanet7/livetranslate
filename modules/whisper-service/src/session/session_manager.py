#!/usr/bin/env python3
"""
Session Management

Manages transcription sessions with persistence and statistics.
Extracted from whisper_service.py for better modularity.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

# Import TranscriptionResult from transcription package
from transcription.request_models import TranscriptionResult

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages transcription sessions with persistence and statistics
    """

    def __init__(self, session_dir: Optional[str] = None):
        """Initialize session manager"""
        self.session_dir = session_dir or os.path.join(os.path.dirname(__file__), "..", "..", "session_data")
        os.makedirs(self.session_dir, exist_ok=True)

        self.sessions: Dict[str, Dict] = {}
        self.transcription_history = deque(maxlen=200)

        # Load existing sessions
        self._load_sessions()

    def create_session(self, session_id: str, config: Optional[Dict] = None) -> Dict:
        """Create a new transcription session"""
        session_config = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "config": config or {},
            "stats": {
                "transcriptions": 0,
                "total_duration": 0.0,
                "total_words": 0,
                "avg_confidence": 0.0
            },
            "transcriptions": []
        }

        self.sessions[session_id] = session_config
        self._save_session(session_id)
        logger.info(f"Created transcription session: {session_id}")
        return session_config

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.sessions.get(session_id)

    def add_transcription(self, session_id: str, result: TranscriptionResult):
        """Add transcription result to session"""
        if session_id not in self.sessions:
            self.create_session(session_id)

        session = self.sessions[session_id]

        # Add to session transcriptions
        transcription_data = {
            "text": result.text,
            "timestamp": result.timestamp,
            "confidence": result.confidence_score,
            "model": result.model_used,
            "device": result.device_used,
            "processing_time": result.processing_time
        }

        session["transcriptions"].append(transcription_data)

        # Update statistics
        stats = session["stats"]
        stats["transcriptions"] += 1
        stats["total_words"] += len(result.text.split())

        # Update average confidence
        old_avg = stats["avg_confidence"]
        count = stats["transcriptions"]
        stats["avg_confidence"] = (old_avg * (count - 1) + result.confidence_score) / count

        # Add to global history
        self.transcription_history.append(transcription_data)

        # Save session
        self._save_session(session_id)

    def close_session(self, session_id: str) -> Optional[Dict]:
        """Close session and return final statistics"""
        session = self.sessions.get(session_id)
        if session:
            session["closed_at"] = datetime.now().isoformat()
            self._save_session(session_id)
            logger.info(f"Closed transcription session: {session_id}")
        return session

    def get_transcription_history(self, limit: int = 50) -> List[Dict]:
        """Get recent transcription history"""
        return list(self.transcription_history)[-limit:]

    def _load_sessions(self):
        """Load sessions from disk"""
        try:
            sessions_file = os.path.join(self.session_dir, "sessions.json")
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sessions = data.get("sessions", {})

            # Load transcription history
            history_file = os.path.join(self.session_dir, "transcriptions.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.transcription_history = deque(data.get("transcriptions", []), maxlen=200)

        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")

    def _save_session(self, session_id: str):
        """Save session to disk"""
        try:
            # Save all sessions
            sessions_file = os.path.join(self.session_dir, "sessions.json")
            sessions_data = {
                "sessions": self.sessions,
                "last_updated": datetime.now().isoformat()
            }

            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)

            # Save transcription history
            history_file = os.path.join(self.session_dir, "transcriptions.json")
            history_data = {
                "transcriptions": list(self.transcription_history)[-100:],
                "last_updated": datetime.now().isoformat()
            }

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
