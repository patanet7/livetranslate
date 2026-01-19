/**
 * CaptionOverlay Component Tests (TDD)
 *
 * Tests for the transparent caption overlay component.
 * Written BEFORE implementation following TDD principles.
 *
 * This component is designed to be used as a Browser Source in OBS
 * for displaying real-time translated captions over video.
 */

import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Mock } from "vitest";

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string): void {}
  close(): void {
    this.readyState = WebSocket.CLOSED;
  }

  // Test helpers
  simulateOpen(): void {
    this.readyState = WebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  simulateMessage(data: object): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }

  simulateClose(): void {
    this.readyState = WebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close"));
  }
}

// Replace global WebSocket
const originalWebSocket = global.WebSocket;

beforeEach(() => {
  MockWebSocket.instances = [];
  global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
});

afterEach(() => {
  global.WebSocket = originalWebSocket;
});

// =============================================================================
// Test Fixtures
// =============================================================================

const sampleCaption = {
  id: "caption-001",
  original_text: "Hello, how are you?",
  translated_text: "Hola, ¿cómo estás?",
  speaker_name: "Alice",
  speaker_color: "#4CAF50",
  target_language: "es",
  timestamp: new Date().toISOString(),
  duration_seconds: 8.0,
  confidence: 0.95,
};

const sampleCaption2 = {
  id: "caption-002",
  original_text: "I am doing well, thanks!",
  translated_text: "¡Estoy bien, gracias!",
  speaker_name: "Bob",
  speaker_color: "#2196F3",
  target_language: "es",
  timestamp: new Date().toISOString(),
  duration_seconds: 6.0,
  confidence: 0.92,
};

// =============================================================================
// Rendering Tests
// =============================================================================

describe("CaptionOverlay", () => {
  describe("Basic Rendering", () => {
    it("renders without crashing", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);
      expect(container).toBeTruthy();
    });

    it("renders with transparent background", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);
      const overlay = container.firstChild as HTMLElement;
      expect(overlay).toBeTruthy();
      // Check for transparent styling
      const style = window.getComputedStyle(overlay);
      expect(style.backgroundColor).toBe("transparent");
    });

    it("fills entire viewport", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);
      const overlay = container.firstChild as HTMLElement;
      expect(overlay.style.width).toBe("100%");
      expect(overlay.style.height).toBe("100%");
    });
  });

  // ===========================================================================
  // WebSocket Connection Tests
  // ===========================================================================

  describe("WebSocket Connection", () => {
    it("connects to caption stream on mount", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(
        <CaptionOverlay sessionId="test-session" wsUrl="ws://localhost:3000" />,
      );

      await waitFor(() => {
        expect(MockWebSocket.instances.length).toBe(1);
        expect(MockWebSocket.instances[0].url).toContain("test-session");
      });
    });

    it("uses default WebSocket URL when not provided", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="my-session" />);

      await waitFor(() => {
        expect(MockWebSocket.instances.length).toBe(1);
        expect(MockWebSocket.instances[0].url).toContain(
          "/api/captions/stream/my-session",
        );
      });
    });

    it("disconnects on unmount", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { unmount } = render(<CaptionOverlay sessionId="test-session" />);

      await waitFor(() => {
        expect(MockWebSocket.instances.length).toBe(1);
      });

      const ws = MockWebSocket.instances[0];
      ws.simulateOpen();

      unmount();

      expect(ws.readyState).toBe(WebSocket.CLOSED);
    });

    it("shows connection status", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" showConnectionStatus />);

      // Initially connecting
      expect(screen.getByText(/connecting/i)).toBeTruthy();

      // Simulate open
      await waitFor(() => {
        MockWebSocket.instances[0].simulateOpen();
      });

      await waitFor(() => {
        expect(screen.getByText(/connected/i)).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Caption Display Tests
  // ===========================================================================

  describe("Caption Display", () => {
    it("displays received captions", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({
          event: "caption_added",
          caption: sampleCaption,
        });
      });

      await waitFor(() => {
        expect(screen.getByText(sampleCaption.translated_text)).toBeTruthy();
      });
    });

    it("displays speaker name with caption", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" showSpeakerName />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({
          event: "caption_added",
          caption: sampleCaption,
        });
      });

      await waitFor(() => {
        expect(screen.getByText(sampleCaption.speaker_name)).toBeTruthy();
      });
    });

    it("applies speaker color to name", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" showSpeakerName />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({
          event: "caption_added",
          caption: sampleCaption,
        });
      });

      await waitFor(() => {
        const speakerElement = screen.getByText(sampleCaption.speaker_name);
        const style = window.getComputedStyle(speakerElement);
        expect(style.color).toBe("rgb(76, 175, 80)"); // #4CAF50
      });
    });

    it("displays original text when enabled", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" showOriginal />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({
          event: "caption_added",
          caption: sampleCaption,
        });
      });

      await waitFor(() => {
        expect(screen.getByText(sampleCaption.original_text)).toBeTruthy();
        expect(screen.getByText(sampleCaption.translated_text)).toBeTruthy();
      });
    });

    it("removes expired captions", async () => {
      vi.useFakeTimers();

      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({
          event: "caption_added",
          caption: { ...sampleCaption, duration_seconds: 1.0 },
        });
      });

      await waitFor(() => {
        expect(screen.getByText(sampleCaption.translated_text)).toBeTruthy();
      });

      // Simulate caption expiration
      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateMessage({
          event: "caption_expired",
          caption_id: sampleCaption.id,
        });
      });

      await waitFor(() => {
        expect(screen.queryByText(sampleCaption.translated_text)).toBeNull();
      });

      vi.useRealTimers();
    });
  });

  // ===========================================================================
  // Multiple Caption Tests
  // ===========================================================================

  describe("Multiple Captions", () => {
    it("displays multiple captions stacked", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" maxCaptions={5} />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption2 });
      });

      await waitFor(() => {
        expect(screen.getByText(sampleCaption.translated_text)).toBeTruthy();
        expect(screen.getByText(sampleCaption2.translated_text)).toBeTruthy();
      });
    });

    it("limits number of visible captions", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" maxCaptions={1} />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption2 });
      });

      await waitFor(() => {
        // Only newest caption visible
        expect(screen.queryByText(sampleCaption.translated_text)).toBeNull();
        expect(screen.getByText(sampleCaption2.translated_text)).toBeTruthy();
      });
    });

    it("assigns different colors to different speakers", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" showSpeakerName />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption2 });
      });

      await waitFor(() => {
        const alice = screen.getByText("Alice");
        const bob = screen.getByText("Bob");

        const aliceStyle = window.getComputedStyle(alice);
        const bobStyle = window.getComputedStyle(bob);

        expect(aliceStyle.color).not.toBe(bobStyle.color);
      });
    });
  });

  // ===========================================================================
  // Styling Configuration Tests
  // ===========================================================================

  describe("Styling Configuration", () => {
    it("respects custom font size", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" fontSize={24} />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
      });

      await waitFor(() => {
        const caption = screen.getByText(sampleCaption.translated_text);
        const style = window.getComputedStyle(caption);
        expect(style.fontSize).toBe("24px");
      });
    });

    it("positions captions at bottom by default", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);

      const overlay = container.firstChild as HTMLElement;
      expect(overlay.style.alignItems).toBe("flex-end");
    });

    it("respects custom position", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(
        <CaptionOverlay sessionId="test-session" position="top" />,
      );

      const overlay = container.firstChild as HTMLElement;
      expect(overlay.style.alignItems).toBe("flex-start");
    });

    it("applies text shadow for readability", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
      });

      await waitFor(() => {
        const caption = screen.getByText(sampleCaption.translated_text);
        const style = window.getComputedStyle(caption);
        expect(style.textShadow).not.toBe("none");
      });
    });

    it("applies custom background style", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(
        <CaptionOverlay
          sessionId="test-session"
          captionBackground="rgba(0,0,0,0.8)"
        />,
      );

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
      });

      await waitFor(() => {
        const caption = screen.getByText(
          sampleCaption.translated_text,
        ).parentElement;
        const style = window.getComputedStyle(caption!);
        expect(style.backgroundColor).toBe("rgba(0, 0, 0, 0.8)");
      });
    });
  });

  // ===========================================================================
  // Animation Tests
  // ===========================================================================

  describe("Animations", () => {
    it("animates caption entry", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      render(<CaptionOverlay sessionId="test-session" animate />);

      await waitFor(() => {
        const ws = MockWebSocket.instances[0];
        ws.simulateOpen();
        ws.simulateMessage({ event: "caption_added", caption: sampleCaption });
      });

      await waitFor(() => {
        const caption = screen.getByText(
          sampleCaption.translated_text,
        ).parentElement;
        const style = window.getComputedStyle(caption!);
        // Check for animation or transition property
        expect(style.animation || style.transition).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Accessibility Tests
  // ===========================================================================

  describe("Accessibility", () => {
    it("has appropriate ARIA labels", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);

      const overlay = container.firstChild as HTMLElement;
      expect(overlay.getAttribute("role")).toBe("region");
      expect(overlay.getAttribute("aria-label")).toContain("captions");
    });

    it("marks captions as live region", async () => {
      const { CaptionOverlay } = await import("../CaptionOverlay");
      const { container } = render(<CaptionOverlay sessionId="test-session" />);

      const liveRegion = container.querySelector("[aria-live]");
      expect(liveRegion).toBeTruthy();
      expect(liveRegion?.getAttribute("aria-live")).toBe("polite");
    });
  });
});
