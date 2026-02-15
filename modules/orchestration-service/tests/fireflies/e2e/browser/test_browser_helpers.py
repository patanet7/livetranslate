"""Smoke test for AgentBrowser wrapper."""

from fireflies.e2e.browser.browser_helpers import AgentBrowser


def test_agent_browser_instantiation():
    """AgentBrowser can be created with default settings."""
    browser = AgentBrowser()
    assert browser.headed is True
    assert browser.stream_port is None


def test_agent_browser_streaming_mode():
    """AgentBrowser can be created in streaming mode."""
    browser = AgentBrowser(headed=False, stream_port=9223)
    assert browser.headed is False
    assert browser.stream_port == 9223
