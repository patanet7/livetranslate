"""TypeScript/Python WebSocket message contract alignment tests.

Parses the TypeScript interface definitions from the dashboard service
and compares them against the Python Pydantic models in ws_messages.py.
Any field mismatch between the two sources indicates a contract drift
that will cause runtime failures.

No mocks -- reads the real TypeScript source file and introspects
real Pydantic model classes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from livetranslate_common.models import ws_messages

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SHARED_TESTS_DIR = Path(__file__).parent
# Path: shared/tests/ -> shared/ -> modules/ -> project root -> modules/dashboard-service/...
_TS_FILE = (
    _SHARED_TESTS_DIR.parent.parent
    / "dashboard-service"
    / "src"
    / "lib"
    / "types"
    / "ws-messages.ts"
)

# ---------------------------------------------------------------------------
# TypeScript parser
# ---------------------------------------------------------------------------


def _parse_ts_interfaces(path: Path) -> dict[str, set[str]]:
    """Extract interface names and their field names from TypeScript source."""
    content = path.read_text()
    interfaces: dict[str, set[str]] = {}
    for match in re.finditer(
        r"export interface (\w+Message)\s*\{([^}]+)\}",
        content,
        re.DOTALL,
    ):
        name = match.group(1)
        body = match.group(2)
        fields: set[str] = set()
        for field_match in re.finditer(r"(\w+)\??:\s", body):
            fields.add(field_match.group(1))
        interfaces[name] = fields
    return interfaces


def _parse_ts_protocol_version(path: Path) -> int | None:
    """Extract PROTOCOL_VERSION constant from TypeScript source."""
    content = path.read_text()
    match = re.search(r"export const PROTOCOL_VERSION\s*=\s*(\d+)", content)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Python ↔ TypeScript mapping
# ---------------------------------------------------------------------------

TS_TO_PYTHON: dict[str, type] = {
    "ConnectedMessage": ws_messages.ConnectedMessage,
    "SegmentMessage": ws_messages.SegmentMessage,
    "InterimMessage": ws_messages.InterimMessage,
    "TranslationMessage": ws_messages.TranslationMessage,
    "MeetingStartedMessage": ws_messages.MeetingStartedMessage,
    "RecordingStatusMessage": ws_messages.RecordingStatusMessage,
    "ServiceStatusMessage": ws_messages.ServiceStatusMessage,
    "LanguageDetectedMessage": ws_messages.LanguageDetectedMessage,
    "BackendSwitchedMessage": ws_messages.BackendSwitchedMessage,
    "StartSessionMessage": ws_messages.StartSessionMessage,
    "EndSessionMessage": ws_messages.EndSessionMessage,
    "PromoteToMeetingMessage": ws_messages.PromoteToMeetingMessage,
    "EndMeetingMessage": ws_messages.EndMeetingMessage,
    "ConfigMessage": ws_messages.ConfigMessage,
    "EndMessage": ws_messages.EndMessage,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTypeScriptPythonAlignment:
    """Verify that TypeScript and Python WebSocket message definitions stay in sync."""

    def test_ts_file_exists(self) -> None:
        assert _TS_FILE.exists(), (
            f"TypeScript WS messages file not found at {_TS_FILE}. "
            "If the dashboard service moved, update the path in this test."
        )

    def test_all_python_messages_have_ts_counterpart(self) -> None:
        ts_interfaces = _parse_ts_interfaces(_TS_FILE)
        ts_names = set(ts_interfaces.keys())

        for type_key, python_cls in ws_messages._ALL_MESSAGES.items():
            class_name = python_cls.__name__
            assert class_name in ts_names, (
                f"Python message class {class_name} (type={type_key!r}) "
                f"has no matching TypeScript interface in {_TS_FILE.name}. "
                f"Available TS interfaces: {sorted(ts_names)}"
            )

    def test_all_ts_messages_have_python_counterpart(self) -> None:
        ts_interfaces = _parse_ts_interfaces(_TS_FILE)
        python_class_names = {cls.__name__ for cls in ws_messages._ALL_MESSAGES.values()}

        for ts_name in ts_interfaces:
            assert ts_name in python_class_names, (
                f"TypeScript interface {ts_name} has no matching Python "
                f"message class in ws_messages.py. "
                f"Available Python classes: {sorted(python_class_names)}"
            )

    def test_field_alignment(self) -> None:
        ts_interfaces = _parse_ts_interfaces(_TS_FILE)
        mismatches: list[str] = []

        for ts_name, python_cls in TS_TO_PYTHON.items():
            if ts_name not in ts_interfaces:
                mismatches.append(f"{ts_name}: not found in TypeScript file")
                continue

            ts_fields = ts_interfaces[ts_name] - {"type"}
            python_fields = set(python_cls.model_fields.keys()) - {"type"}

            if ts_fields != python_fields:
                only_ts = ts_fields - python_fields
                only_py = python_fields - ts_fields
                parts = [f"{ts_name}:"]
                if only_ts:
                    parts.append(f"  only in TS: {sorted(only_ts)}")
                if only_py:
                    parts.append(f"  only in Python: {sorted(only_py)}")
                mismatches.append("\n".join(parts))

        assert not mismatches, (
            "Field mismatches between TypeScript and Python message definitions:\n"
            + "\n".join(mismatches)
        )

    def test_protocol_version_matches(self) -> None:
        ts_version = _parse_ts_protocol_version(_TS_FILE)
        py_version = ws_messages.PROTOCOL_VERSION

        assert ts_version is not None, (
            "Could not find PROTOCOL_VERSION in TypeScript file. "
            "Expected: export const PROTOCOL_VERSION = <number>"
        )
        assert ts_version == py_version, (
            f"Protocol version mismatch: TypeScript={ts_version}, Python={py_version}. "
            "Update both files to the same version before deploying."
        )

    def test_ts_parser_extracts_known_count(self) -> None:
        """Sanity check: the regex parser finds the expected number of interfaces."""
        ts_interfaces = _parse_ts_interfaces(_TS_FILE)
        # There are 15 message types in _ALL_MESSAGES
        expected_count = len(ws_messages._ALL_MESSAGES)
        assert len(ts_interfaces) == expected_count, (
            f"Expected {expected_count} TS interfaces, found {len(ts_interfaces)}. "
            f"Found: {sorted(ts_interfaces.keys())}"
        )
