"""Tests for MeetingSessionConfig — thread-safe, observable config."""

import threading

import pytest

from services.meeting_session_config import MeetingSessionConfig


class TestMeetingSessionConfig:
    def test_create_with_defaults(self):
        config = MeetingSessionConfig(session_id="test-123")
        assert config.session_id == "test-123"
        assert config.caption_source == "bot_audio"
        assert config.source_lang == "auto"
        assert config.target_lang == "en"
        assert config.display_mode == "subtitle"
        assert config.theme == "dark"
        assert config.font_size == 24
        assert config.show_speakers is True
        assert config.show_original is False
        assert config.translation_enabled is True

    def test_update_returns_changed_fields(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(target_lang="zh", font_size=32)
        assert changed == {"target_lang", "font_size"}
        assert config.target_lang == "zh"
        assert config.font_size == 32

    def test_update_no_change_returns_empty(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(target_lang="en")  # Same as default
        assert changed == set()

    def test_subscriber_notified_on_change(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        config.subscribe(lambda fields: notifications.append(fields))

        config.update(target_lang="zh")
        assert len(notifications) == 1
        assert notifications[0] == {"target_lang"}

    def test_subscriber_not_notified_when_no_change(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        config.subscribe(lambda fields: notifications.append(fields))

        config.update(target_lang="en")  # Same as default
        assert len(notifications) == 0

    def test_multiple_subscribers(self):
        config = MeetingSessionConfig(session_id="test-123")
        notif_a, notif_b = [], []
        config.subscribe(lambda f: notif_a.append(f))
        config.subscribe(lambda f: notif_b.append(f))

        config.update(theme="light")
        assert len(notif_a) == 1
        assert len(notif_b) == 1

    def test_unsubscribe(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        callback = lambda f: notifications.append(f)
        config.subscribe(callback)
        config.unsubscribe(callback)

        config.update(target_lang="zh")
        assert len(notifications) == 0

    def test_snapshot_returns_frozen_copy(self):
        config = MeetingSessionConfig(session_id="test-123")
        snap = config.snapshot()
        config.update(target_lang="zh")
        assert snap["target_lang"] == "en"  # Snapshot unchanged
        assert config.target_lang == "zh"   # Config changed

    def test_batch_update_single_notification(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        config.subscribe(lambda f: notifications.append(f))

        config.update(target_lang="zh", font_size=32, theme="light")
        assert len(notifications) == 1  # ONE notification for batch
        assert notifications[0] == {"target_lang", "font_size", "theme"}

    def test_ignores_unknown_fields(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(nonexistent_field="value")
        assert changed == set()

    def test_ignores_private_fields(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(_lock="hacked")
        assert changed == set()

    def test_thread_safety_concurrent_updates(self):
        config = MeetingSessionConfig(session_id="test-123")
        errors = []

        def updater(lang: str):
            try:
                for _ in range(100):
                    config.update(target_lang=lang)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=updater, args=("zh",)),
            threading.Thread(target=updater, args=("en",)),
            threading.Thread(target=updater, args=("ja",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert config.target_lang in ("zh", "en", "ja")
