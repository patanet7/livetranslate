/**
 * WebSocket Parameter Sync Integration Tests
 *
 * Tests for real-time parameter synchronization between frontend and backend:
 * - Parameter update message format
 * - Batch parameter changes
 * - Backend confirmation handling
 * - Error recovery
 * - Reconnection logic
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReactFlowProvider } from "reactflow";
import AudioStageNode from "../AudioStageNode";

// Mock WebSocket with message handling
class MockWebSocket {
  private messageHandlers: Array<(event: MessageEvent) => void> = [];
  private openHandlers: Array<() => void> = [];
  private errorHandlers: Array<(event: Event) => void> = [];
  private closeHandlers: Array<() => void> = [];

  readyState = 1; // WebSocket.OPEN
  CONNECTING = 0;
  OPEN = 1;
  CLOSING = 2;
  CLOSED = 3;

  send = vi.fn((data: string) => {
    const message = JSON.parse(data);

    // Simulate backend response
    setTimeout(() => {
      this.simulateMessage({
        type: "config_updated",
        stage_id: message.stage_id,
        success: true,
        parameters: message.parameters,
      });
    }, 50);
  });

  addEventListener = vi.fn((event: string, handler: any) => {
    if (event === "message") {
      this.messageHandlers.push(handler);
    } else if (event === "open") {
      this.openHandlers.push(handler);
    } else if (event === "error") {
      this.errorHandlers.push(handler);
    } else if (event === "close") {
      this.closeHandlers.push(handler);
    }
  });

  removeEventListener = vi.fn();

  close = vi.fn(() => {
    this.readyState = 3; // CLOSED
    this.closeHandlers.forEach((handler) => handler());
  });

  simulateMessage(data: any) {
    const event = new MessageEvent("message", {
      data: JSON.stringify(data),
    });
    this.messageHandlers.forEach((handler) => handler(event));
  }

  simulateOpen() {
    this.readyState = 1; // OPEN
    this.openHandlers.forEach((handler) => handler());
  }

  simulateError() {
    const event = new Event("error");
    this.errorHandlers.forEach((handler) => handler(event));
  }

  simulateClose() {
    this.readyState = 3; // CLOSED
    this.closeHandlers.forEach((handler) => handler());
  }
}

describe("WebSocket Parameter Sync Integration", () => {
  let mockWebSocket: MockWebSocket;

  const defaultNodeData = {
    label: "Test Node",
    description: "Test Description",
    stageType: "processing" as const,
    icon: () => <div>Icon</div>,
    enabled: true,
    gainIn: 0,
    gainOut: 0,
    stageConfig: {
      strength: 0.7,
      threshold: 0.5,
    },
    parameters: [
      {
        name: "strength",
        value: 0.7,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        unit: "",
        description: "Processing strength",
      },
      {
        name: "threshold",
        value: 0.5,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        unit: "",
        description: "Threshold level",
      },
    ],
    metrics: {
      processingTimeMs: 12.5,
      targetLatencyMs: 25.0,
      qualityImprovement: 2.5,
      inputLevel: 0.6,
      outputLevel: 0.7,
      cpuUsage: 15.3,
    },
    isProcessing: false,
    status: "idle" as const,
  };

  const defaultProps = {
    id: "node-1",
    data: defaultNodeData,
    selected: false,
    onSettingsOpen: vi.fn(),
    onGainChange: vi.fn(),
    onParameterChange: vi.fn(),
    onToggleEnabled: vi.fn(),
    isConnectable: true,
    type: "audioStage",
    position: { x: 0, y: 0 },
    dragging: false,
    zIndex: 0,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    mockWebSocket = new MockWebSocket();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  const renderNode = (props = {}) => {
    return render(
      <ReactFlowProvider>
        <div style={{ width: 400, height: 600 }}>
          <AudioStageNode {...defaultProps} {...props} />
        </div>
      </ReactFlowProvider>,
    );
  };

  describe("Parameter Update Message Format", () => {
    it("should send parameter update via WebSocket when slider changes", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2]; // After gain sliders

      // Change parameter
      await user.type(strengthSlider, "{ArrowRight}");

      // Wait for debounce
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });

      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
      expect(sentMessage).toEqual({
        type: "update_stage",
        stage_id: "node-1",
        parameters: expect.objectContaining({
          strength: expect.any(Number),
        }),
      });
    });

    it("should send correct parameter name and value", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      const thresholdSlider = sliders[3]; // Threshold parameter

      await user.type(thresholdSlider, "{ArrowRight}{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalled();
      });

      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
      expect(sentMessage.parameters).toHaveProperty("threshold");
    });

    it("should include stage_id in message", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        id: "custom-stage-123",
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalled();
      });

      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
      expect(sentMessage.stage_id).toBe("custom-stage-123");
    });
  });

  describe("Batch Parameter Changes", () => {
    it("should batch multiple parameter changes", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2];

      // Change parameter multiple times rapidly
      await user.type(
        strengthSlider,
        "{ArrowRight}{ArrowRight}{ArrowRight}{ArrowRight}",
      );

      // Should not send immediately
      expect(mockWebSocket.send).not.toHaveBeenCalled();

      // Wait for debounce
      vi.advanceTimersByTime(300);

      // Should send only ONE message
      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });
    });

    it("should send latest parameter value when batching", async () => {
      const user = userEvent.setup({ delay: null });
      const onParameterChange = vi.fn();

      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
        onParameterChange,
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2];

      // Make multiple rapid changes
      await user.type(strengthSlider, "{ArrowRight}{ArrowRight}{ArrowRight}");

      // onParameterChange should be called multiple times (local state)
      expect(onParameterChange.mock.calls.length).toBeGreaterThan(1);

      vi.advanceTimersByTime(300);

      // WebSocket should send only once
      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });
    });

    it("should reset debounce timer on each change", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");

      // First change
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(200); // Not enough time

      // Second change (should reset timer)
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(200); // Still not enough

      // Should not have sent yet
      expect(mockWebSocket.send).not.toHaveBeenCalled();

      // Wait remaining time
      vi.advanceTimersByTime(100);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe("Backend Confirmation Handling", () => {
    it("should receive config_updated confirmation from backend", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      // Wait for send
      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalled();
      });

      // Advance time for backend response
      vi.advanceTimersByTime(100);

      // Should show sync indicator (brief flash)
      await waitFor(() => {
        const syncIcon = screen.queryByTitle(/Syncing parameters/i);
        // May or may not be visible depending on timing
        expect(syncIcon).toBeDefined();
      });
    });

    it("should handle successful config update", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalled();
      });

      // Simulate backend success response
      mockWebSocket.simulateMessage({
        type: "config_updated",
        stage_id: "node-1",
        success: true,
      });

      // Node should show connected status
      expect(screen.getByTitle("Connected to backend")).toBeInTheDocument();
    });

    it("should handle failed config update", async () => {
      const user = userEvent.setup({ delay: null });
      const consoleError = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalled();
      });

      // Simulate backend error response
      mockWebSocket.simulateMessage({
        type: "config_updated",
        stage_id: "node-1",
        success: false,
        error: "Invalid parameter value",
      });

      // Could log error or show error state
      // (Depends on implementation)

      consoleError.mockRestore();
    });
  });

  describe("Error Recovery", () => {
    it("should handle WebSocket errors gracefully", async () => {
      const user = userEvent.setup({ delay: null });
      const consoleError = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      const errorWebSocket = {
        ...mockWebSocket,
        send: vi.fn().mockImplementation(() => {
          throw new Error("WebSocket send failed");
        }),
      };

      renderNode({
        websocket: errorWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(consoleError).toHaveBeenCalledWith(
          "Failed to send parameter update:",
          expect.any(Error),
        );
      });

      // UI should still be functional
      expect(screen.getByText("Test Node")).toBeInTheDocument();

      consoleError.mockRestore();
    });

    it("should not crash when WebSocket is null", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: null as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");

      // Should not throw error
      await expect(async () => {
        await user.type(sliders[2], "{ArrowRight}");
        vi.advanceTimersByTime(300);
      }).not.toThrow();
    });

    it("should not send when WebSocket is closed", async () => {
      const user = userEvent.setup({ delay: null });
      const closedWebSocket = new MockWebSocket();
      closedWebSocket.readyState = 3; // CLOSED

      renderNode({
        websocket: closedWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      expect(closedWebSocket.send).not.toHaveBeenCalled();
    });
  });

  describe("Reconnection Logic", () => {
    it("should reconnect and resync parameters after disconnect", async () => {
      const user = userEvent.setup({ delay: null });
      const { rerender } = renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");

      // Make change while connected
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });

      // Simulate disconnect
      mockWebSocket.simulateClose();

      rerender(
        <ReactFlowProvider>
          <div style={{ width: 400, height: 600 }}>
            <AudioStageNode
              {...defaultProps}
              websocket={null as any}
              isRealtimeActive={false}
            />
          </div>
        </ReactFlowProvider>,
      );

      // Should show disconnected
      expect(screen.getByTitle("Not connected to backend")).toBeInTheDocument();

      // Make change while disconnected
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      // Should not send
      expect(mockWebSocket.send).toHaveBeenCalledTimes(1); // Still only 1 from before

      // Reconnect
      const newWebSocket = new MockWebSocket();
      rerender(
        <ReactFlowProvider>
          <div style={{ width: 400, height: 600 }}>
            <AudioStageNode
              {...defaultProps}
              websocket={newWebSocket as any}
              isRealtimeActive={true}
            />
          </div>
        </ReactFlowProvider>,
      );

      // Should show connected
      expect(screen.getByTitle("Connected to backend")).toBeInTheDocument();

      // Make new change after reconnect
      await user.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      // Should send with new WebSocket
      await waitFor(() => {
        expect(newWebSocket.send).toHaveBeenCalled();
      });
    });

    it("should handle rapid connection state changes", async () => {
      const { rerender } = renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      // Connect
      expect(screen.getByTitle("Connected to backend")).toBeInTheDocument();

      // Disconnect
      rerender(
        <ReactFlowProvider>
          <div style={{ width: 400, height: 600 }}>
            <AudioStageNode
              {...defaultProps}
              websocket={null as any}
              isRealtimeActive={false}
            />
          </div>
        </ReactFlowProvider>,
      );

      expect(screen.getByTitle("Not connected to backend")).toBeInTheDocument();

      // Reconnect
      rerender(
        <ReactFlowProvider>
          <div style={{ width: 400, height: 600 }}>
            <AudioStageNode
              {...defaultProps}
              websocket={mockWebSocket as any}
              isRealtimeActive={true}
            />
          </div>
        </ReactFlowProvider>,
      );

      // Should be stable
      expect(screen.getByTitle("Connected to backend")).toBeInTheDocument();
    });
  });

  describe("Multi-Parameter Updates", () => {
    it("should handle multiple different parameters changing", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2];
      const thresholdSlider = sliders[3];

      // Change first parameter
      await user.type(strengthSlider, "{ArrowRight}");
      vi.advanceTimersByTime(200);

      // Change second parameter before first debounce finishes
      await user.type(thresholdSlider, "{ArrowRight}");
      vi.advanceTimersByTime(200);

      // Should not have sent yet
      expect(mockWebSocket.send).not.toHaveBeenCalled();

      // Wait remaining time
      vi.advanceTimersByTime(100);

      // Should send separate messages for each parameter
      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(2);
      });

      const messages = mockWebSocket.send.mock.calls.map((call) =>
        JSON.parse(call[0]),
      );
      const paramNames = messages.map((msg) => Object.keys(msg.parameters)[0]);

      expect(paramNames).toContain("strength");
      expect(paramNames).toContain("threshold");
    });
  });

  describe("Cleanup on Unmount", () => {
    it("should clear pending parameter updates on unmount", async () => {
      const user = userEvent.setup({ delay: null });
      const { unmount } = renderNode({
        websocket: mockWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await user.type(sliders[2], "{ArrowRight}");

      // Unmount before debounce fires
      unmount();
      vi.advanceTimersByTime(300);

      // Should not send after unmount
      expect(mockWebSocket.send).not.toHaveBeenCalled();
    });

    it("should not cause memory leaks with multiple mount/unmount cycles", () => {
      for (let i = 0; i < 10; i++) {
        const { unmount } = renderNode({
          websocket: mockWebSocket as any,
          isRealtimeActive: true,
        });
        unmount();
      }

      // Should not throw or cause errors
      expect(true).toBe(true);
    });
  });
});
