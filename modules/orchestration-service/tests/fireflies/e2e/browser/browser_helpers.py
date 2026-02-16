"""
AgentBrowser — Python wrapper around agent-browser CLI.

Calls agent-browser commands via subprocess. Used by pytest to drive
Chromium for browser-level E2E testing of the Fireflies dashboard.
"""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

# How long to wait for agent-browser commands (seconds)
DEFAULT_TIMEOUT = 30


class AgentBrowserError(Exception):
    """Raised when an agent-browser command fails."""


class AgentBrowser:
    """
    Thin Python wrapper around the agent-browser CLI.

    Usage:
        browser = AgentBrowser()
        browser.open("http://localhost:3001/static/fireflies-dashboard.html")
        snapshot = browser.snapshot(interactive=True)
        browser.click("@e5")
        browser.fill("@e3", "my-api-key")
        browser.screenshot("/tmp/test.png")
        browser.close()
    """

    def __init__(
        self,
        headed: bool = True,
        stream_port: int | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.headed = headed
        self.stream_port = stream_port
        self.timeout = timeout
        self._started = False

    def _run(self, *args: str, timeout: int | None = None) -> str:
        """Run an agent-browser CLI command and return stdout."""
        cmd = ["agent-browser"]
        if self.headed:
            cmd.append("--headed")
        cmd.extend(args)
        env = os.environ.copy()

        if self.stream_port and not self.headed:
            env["AGENT_BROWSER_STREAM_PORT"] = str(self.stream_port)

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise AgentBrowserError(
                f"agent-browser command timed out after {timeout or self.timeout}s: {' '.join(cmd)}"
            ) from e

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise AgentBrowserError(
                f"agent-browser command failed (exit {result.returncode}): {stderr}"
            )

        return result.stdout.strip()

    # =========================================================================
    # Navigation
    # =========================================================================

    def open(self, url: str) -> str:
        """Navigate to a URL. Starts browser if not started."""
        output = self._run("open", url)
        self._started = True
        return output

    def reload(self) -> str:
        """Reload the current page."""
        return self._run("reload")

    # =========================================================================
    # Inspection
    # =========================================================================

    def snapshot(
        self, interactive: bool = False, compact: bool = False, selector: str | None = None
    ) -> str:
        """
        Get accessibility tree snapshot of the page.

        Args:
            interactive: Only include interactive elements (@refs)
            compact: Remove empty structural nodes
            selector: CSS selector to scope the snapshot

        Returns:
            Accessibility tree text with @ref annotations.
        """
        args = ["snapshot"]
        if interactive:
            args.append("-i")
        if compact:
            args.append("-c")
        if selector:
            args.extend(["-s", selector])
        return self._run(*args)

    def screenshot(self, path: str) -> str:
        """Take a screenshot and save to path."""
        return self._run("screenshot", path)

    def get_text(self, selector: str) -> str:
        """Get text content of an element by CSS selector or @ref."""
        return self._run("get", "text", selector)

    def get_html(self, selector: str) -> str:
        """Get inner HTML of an element."""
        return self._run("get", "html", selector)

    def get_value(self, selector: str) -> str:
        """Get value of an input element."""
        return self._run("get", "value", selector)

    def get_attr(self, selector: str, attr_name: str) -> str:
        """Get an attribute value of an element."""
        return self._run("get", "attr", selector, attr_name)

    def get_count(self, selector: str) -> int:
        """Count elements matching a CSS selector."""
        output = self._run("get", "count", selector)
        return int(output.strip())

    def get_styles(self, selector: str) -> str:
        """Get computed styles of an element."""
        return self._run("get", "styles", selector)

    def get_url(self) -> str:
        """Get current page URL."""
        return self._run("get", "url")

    def get_title(self) -> str:
        """Get page title."""
        return self._run("get", "title")

    def is_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        try:
            output = self._run("is", "visible", selector)
            return "true" in output.lower()
        except AgentBrowserError:
            return False

    # =========================================================================
    # Interaction
    # =========================================================================

    def click(self, selector: str) -> str:
        """Click an element by CSS selector or @ref."""
        return self._run("click", selector)

    def fill(self, selector: str, text: str) -> str:
        """Clear and fill an input element."""
        return self._run("fill", selector, text)

    def type_text(self, selector: str, text: str) -> str:
        """Type text into an element (appends, doesn't clear)."""
        return self._run("type", selector, text)

    def select(self, selector: str, *values: str) -> str:
        """Select dropdown option(s)."""
        return self._run("select", selector, *values)

    def press(self, key: str) -> str:
        """Press a keyboard key."""
        return self._run("press", key)

    def scroll(self, direction: str, pixels: int | None = None) -> str:
        """Scroll the page."""
        args = ["scroll", direction]
        if pixels is not None:
            args.append(str(pixels))
        return self._run(*args)

    # =========================================================================
    # JavaScript
    # =========================================================================

    def eval_js(self, js: str) -> str:
        """Evaluate JavaScript in the page context."""
        return self._run("eval", js)

    # =========================================================================
    # Waiting
    # =========================================================================

    def wait(self, selector_or_ms: str | int) -> str:
        """Wait for an element to appear or a number of milliseconds."""
        return self._run("wait", str(selector_or_ms))

    def wait_for_text(self, text: str, timeout: float = 10.0, poll_interval: float = 0.5) -> bool:
        """
        Poll the page until the given text appears in the snapshot.

        Args:
            text: The text to search for.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.

        Returns:
            True if text found within timeout, False otherwise.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                snap = self.snapshot()
                if text in snap:
                    return True
            except AgentBrowserError:
                pass
            time.sleep(poll_interval)
        return False

    def wait_for_element(self, selector: str, timeout: float = 10.0) -> bool:
        """
        Wait for an element matching the CSS selector to exist.

        Args:
            selector: CSS selector.
            timeout: Maximum seconds to wait.

        Returns:
            True if element found, False if timed out.
        """
        try:
            self.wait(selector)
            return True
        except AgentBrowserError:
            return False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the browser."""
        if self._started:
            try:
                self._run("close", timeout=10)
            except AgentBrowserError:
                pass  # Browser may already be closed
            self._started = False
