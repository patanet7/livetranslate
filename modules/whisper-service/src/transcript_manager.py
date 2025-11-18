#!/usr/bin/env python3
"""
Transcript Manager for Whisper Service

Manages complete conversation transcripts with session-based storage,
text deduplication tracking, and conversation history management.

Features:
- Session-based transcript storage
- Complete conversation history
- Text segment tracking with metadata
- Export capabilities for different formats
- Memory-efficient storage with configurable retention
- Integration with continuous stream processor
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with metadata."""
    text: str
    timestamp: float
    session_id: str
    language: str = 'unknown'
    confidence: float = 1.0
    inference_number: int = 0
    speaker_id: Optional[str] = None
    is_complete_sentence: bool = True
    segment_type: str = 'transcription'  # 'transcription', 'translation', 'correction'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SessionStats:
    """Statistics for a transcript session."""
    session_id: str
    created_at: float
    last_updated: float
    total_segments: int = 0
    total_characters: int = 0
    languages_detected: set = None
    avg_confidence: float = 0.0
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.languages_detected is None:
            self.languages_detected = set()

class TranscriptManager:
    """
    Manages complete conversation transcripts with session-based organization.
    
    Provides storage, retrieval, and export of complete transcription sessions
    with proper deduplication tracking and conversation history management.
    """
    
    def __init__(self, storage_dir: str = None, max_sessions: int = 100, 
                 session_timeout: float = 3600.0):
        """
        Initialize transcript manager.
        
        Args:
            storage_dir: Directory for persistent storage (optional)
            max_sessions: Maximum number of active sessions to keep in memory
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.storage_dir = storage_dir
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        
        # In-memory session storage
        self.sessions: Dict[str, List[TranscriptSegment]] = {}
        self.session_stats: Dict[str, SessionStats] = {}
        self.session_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Global lock for session management
        self.manager_lock = threading.RLock()
        
        # Background cleanup tracking
        self.last_cleanup = time.time()
        self.cleanup_interval = 300.0  # 5 minutes
        
        # Setup persistent storage if directory provided
        if self.storage_dir:
            self.storage_path = Path(self.storage_dir)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Transcript storage initialized: {self.storage_path}")
        else:
            self.storage_path = None
            logger.info("Transcript manager initialized (memory-only)")
        
        logger.info("TranscriptManager initialized:")
        logger.info(f"  Max sessions: {self.max_sessions}")
        logger.info(f"  Session timeout: {self.session_timeout}s")
        logger.info(f"  Storage: {'persistent' if self.storage_path else 'memory-only'}")
    
    def store_segment(self, text: str, session_id: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store a transcript segment for a session.
        
        Args:
            text: Transcribed text segment
            session_id: Session identifier
            metadata: Additional metadata about the segment
            
        Returns:
            True if segment was stored successfully
        """
        if not text or not text.strip():
            return False
        
        try:
            metadata = metadata or {}
            current_time = time.time()
            
            # Create transcript segment
            segment = TranscriptSegment(
                text=text.strip(),
                timestamp=current_time,
                session_id=session_id,
                language=metadata.get('language', 'unknown'),
                confidence=metadata.get('confidence', 1.0),
                inference_number=metadata.get('inference_number', 0),
                speaker_id=metadata.get('speaker_id'),
                is_complete_sentence=metadata.get('is_complete_sentence', True),
                segment_type=metadata.get('segment_type', 'transcription'),
                metadata=metadata
            )
            
            with self.session_locks[session_id]:
                # Initialize session if new
                if session_id not in self.sessions:
                    self._create_session(session_id, current_time)
                
                # Add segment to session
                self.sessions[session_id].append(segment)
                
                # Update session statistics
                self._update_session_stats(session_id, segment)
                
                logger.debug(f"[{session_id}] Stored segment: '{text[:50]}...' "
                           f"(lang: {segment.language}, conf: {segment.confidence:.2f})")
            
            # Periodic cleanup
            self._maybe_run_cleanup()
            
            # Persist if storage enabled
            if self.storage_path:
                self._persist_segment(segment)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store transcript segment: {e}")
            return False
    
    def get_session_transcript(self, session_id: str, format: str = 'text') -> Optional[str]:
        """
        Get complete transcript for a session.
        
        Args:
            session_id: Session identifier
            format: Output format ('text', 'json', 'srt')
            
        Returns:
            Formatted transcript or None if session not found
        """
        try:
            with self.session_locks[session_id]:
                if session_id not in self.sessions:
                    return None
                
                segments = self.sessions[session_id]
                
                if format.lower() == 'text':
                    return self._format_as_text(segments)
                elif format.lower() == 'json':
                    return self._format_as_json(segments)
                elif format.lower() == 'srt':
                    return self._format_as_srt(segments)
                else:
                    logger.warning(f"Unsupported format: {format}")
                    return self._format_as_text(segments)
                    
        except Exception as e:
            logger.error(f"Failed to get session transcript: {e}")
            return None
    
    def get_session_segments(self, session_id: str, limit: int = None) -> List[TranscriptSegment]:
        """
        Get transcript segments for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of segments to return (newest first)
            
        Returns:
            List of transcript segments
        """
        try:
            with self.session_locks[session_id]:
                if session_id not in self.sessions:
                    return []
                
                segments = self.sessions[session_id]
                
                if limit is None:
                    return segments.copy()
                else:
                    return segments[-limit:] if limit > 0 else segments[:abs(limit)]
                    
        except Exception as e:
            logger.error(f"Failed to get session segments: {e}")
            return []
    
    def get_session_stats(self, session_id: str) -> Optional[SessionStats]:
        """Get statistics for a session."""
        try:
            with self.session_locks[session_id]:
                return self.session_stats.get(session_id)
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return None
    
    def get_recent_context(self, session_id: str, max_segments: int = 5, 
                          max_age_seconds: float = 300.0) -> List[TranscriptSegment]:
        """
        Get recent transcript segments for context.
        
        Args:
            session_id: Session identifier
            max_segments: Maximum number of segments to return
            max_age_seconds: Maximum age of segments in seconds
            
        Returns:
            List of recent transcript segments
        """
        try:
            with self.session_locks[session_id]:
                if session_id not in self.sessions:
                    return []
                
                current_time = time.time()
                segments = self.sessions[session_id]
                
                # Filter by age and limit
                recent_segments = []
                for segment in reversed(segments):  # Start from newest
                    if current_time - segment.timestamp <= max_age_seconds:
                        recent_segments.append(segment)
                        if len(recent_segments) >= max_segments:
                            break
                    else:
                        break  # Segments are ordered by time
                
                return list(reversed(recent_segments))  # Return in chronological order
                
        except Exception as e:
            logger.error(f"Failed to get recent context: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all data for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared successfully
        """
        try:
            with self.manager_lock:
                # Archive session if storage enabled
                if self.storage_path and session_id in self.sessions:
                    self._archive_session(session_id)
                
                # Remove from memory
                self.sessions.pop(session_id, None)
                self.session_stats.pop(session_id, None)
                
                # Clean up lock
                if session_id in self.session_locks:
                    del self.session_locks[session_id]
                
                logger.info(f"Session {session_id} cleared")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def list_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self.manager_lock:
            return list(self.sessions.keys())
    
    def get_total_stats(self) -> Dict[str, Any]:
        """Get overall statistics across all sessions."""
        try:
            with self.manager_lock:
                total_sessions = len(self.sessions)
                total_segments = sum(len(segments) for segments in self.sessions.values())
                total_characters = sum(
                    stats.total_characters for stats in self.session_stats.values()
                )
                
                all_languages = set()
                for stats in self.session_stats.values():
                    all_languages.update(stats.languages_detected)
                
                return {
                    'total_sessions': total_sessions,
                    'total_segments': total_segments,
                    'total_characters': total_characters,
                    'languages_detected': list(all_languages),
                    'storage_enabled': self.storage_path is not None,
                    'session_timeout': self.session_timeout,
                    'last_cleanup': self.last_cleanup
                }
                
        except Exception as e:
            logger.error(f"Failed to get total stats: {e}")
            return {}
    
    def export_session(self, session_id: str, filepath: str, format: str = 'json') -> bool:
        """
        Export session transcript to file.
        
        Args:
            session_id: Session identifier
            filepath: Output file path
            format: Export format ('json', 'text', 'srt')
            
        Returns:
            True if export was successful
        """
        try:
            transcript = self.get_session_transcript(session_id, format)
            if transcript is None:
                logger.error(f"Session {session_id} not found")
                return False
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            logger.info(f"Session {session_id} exported to {filepath} ({format})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return False
    
    def _create_session(self, session_id: str, timestamp: float):
        """Create a new session."""
        self.sessions[session_id] = []
        self.session_stats[session_id] = SessionStats(
            session_id=session_id,
            created_at=timestamp,
            last_updated=timestamp,
            languages_detected=set()
        )
        logger.info(f"Created new transcript session: {session_id}")
    
    def _update_session_stats(self, session_id: str, segment: TranscriptSegment):
        """Update statistics for a session."""
        stats = self.session_stats[session_id]
        stats.total_segments += 1
        stats.total_characters += len(segment.text)
        stats.languages_detected.add(segment.language)
        stats.last_updated = segment.timestamp
        stats.duration_seconds = segment.timestamp - stats.created_at
        
        # Update average confidence
        if stats.total_segments == 1:
            stats.avg_confidence = segment.confidence
        else:
            old_avg = stats.avg_confidence
            count = stats.total_segments
            stats.avg_confidence = (old_avg * (count - 1) + segment.confidence) / count
    
    def _format_as_text(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as plain text."""
        lines = []
        for segment in segments:
            timestamp_str = datetime.fromtimestamp(segment.timestamp).strftime("%H:%M:%S")
            speaker_str = f"[{segment.speaker_id}] " if segment.speaker_id else ""
            lines.append(f"[{timestamp_str}] {speaker_str}{segment.text}")
        
        return "\n".join(lines)
    
    def _format_as_json(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as JSON."""
        data = {
            'session_id': segments[0].session_id if segments else '',
            'total_segments': len(segments),
            'segments': [asdict(segment) for segment in segments]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _format_as_srt(self, segments: List[TranscriptSegment]) -> str:
        """Format segments as SRT subtitle format."""
        lines = []
        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_time(segment.timestamp)
            # Estimate end time (1 second per segment or next segment time)
            if i < len(segments):
                end_time = self._format_srt_time(segments[i].timestamp)
            else:
                end_time = self._format_srt_time(segment.timestamp + 1.0)
            
            lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                segment.text,
                ""
            ])
        
        return "\n".join(lines)
    
    def _format_srt_time(self, timestamp: float) -> str:
        """Format timestamp for SRT format."""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%H:%M:%S,000")
    
    def _persist_segment(self, segment: TranscriptSegment):
        """Persist segment to storage."""
        if not self.storage_path:
            return
        
        try:
            # Create session directory
            session_dir = self.storage_path / segment.session_id
            session_dir.mkdir(exist_ok=True)
            
            # Append to session file
            session_file = session_dir / "transcript.jsonl"
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to persist segment: {e}")
    
    def _archive_session(self, session_id: str):
        """Archive a complete session to storage."""
        if not self.storage_path or session_id not in self.sessions:
            return
        
        try:
            session_dir = self.storage_path / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save complete transcript
            transcript_file = session_dir / "complete_transcript.json"
            segments = self.sessions[session_id]
            stats = self.session_stats.get(session_id)
            
            data = {
                'session_id': session_id,
                'stats': asdict(stats) if stats else {},
                'segments': [asdict(segment) for segment in segments],
                'archived_at': time.time()
            }
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Session {session_id} archived to {transcript_file}")
            
        except Exception as e:
            logger.warning(f"Failed to archive session {session_id}: {e}")
    
    def _maybe_run_cleanup(self):
        """Run cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            self._cleanup_old_sessions()
            self.last_cleanup = current_time
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions based on timeout and memory limits."""
        try:
            with self.manager_lock:
                current_time = time.time()
                sessions_to_remove = []
                
                # Find sessions to remove (old or excess)
                for session_id, stats in self.session_stats.items():
                    if current_time - stats.last_updated > self.session_timeout:
                        sessions_to_remove.append(session_id)
                
                # Remove oldest sessions if we exceed max_sessions
                if len(self.sessions) > self.max_sessions:
                    sorted_sessions = sorted(
                        self.session_stats.items(),
                        key=lambda x: x[1].last_updated
                    )
                    excess_count = len(self.sessions) - self.max_sessions
                    for session_id, _ in sorted_sessions[:excess_count]:
                        if session_id not in sessions_to_remove:
                            sessions_to_remove.append(session_id)
                
                # Remove sessions
                for session_id in sessions_to_remove:
                    self.clear_session(session_id)
                
                if sessions_to_remove:
                    logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
                    
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    def shutdown(self):
        """Shutdown transcript manager and save all sessions."""
        logger.info("Shutting down TranscriptManager")
        try:
            # Archive all active sessions
            if self.storage_path:
                for session_id in list(self.sessions.keys()):
                    self._archive_session(session_id)
            
            # Clear memory
            with self.manager_lock:
                self.sessions.clear()
                self.session_stats.clear()
                self.session_locks.clear()
            
            logger.info("✓ TranscriptManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during transcript manager shutdown: {e}")


# Factory functions
def create_transcript_manager(storage_dir: str = None, **kwargs) -> TranscriptManager:
    """Create a transcript manager with optional persistent storage."""
    return TranscriptManager(storage_dir=storage_dir, **kwargs)

def create_memory_transcript_manager(**kwargs) -> TranscriptManager:
    """Create a memory-only transcript manager."""
    return TranscriptManager(storage_dir=None, **kwargs)