/**
 * AudioStageNode Component Tests
 *
 * Tests for audio stage node interactions including:
 * - Parameter slider changes
 * - WebSocket parameter broadcasting
 * - Enable/disable toggling
 * - Gain control adjustments
 * - Real-time sync indicators
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReactFlowProvider } from "reactflow";
import AudioStageNode from "../AudioStageNode";

// Mock Material-UI components that cause issues in tests
vi.mock("@mui/material/Slider", () => ({
  default: ({ value, onChange, min, max, ...props }: any) => (
    <input
      type="range"
      value={value}
      onChange={(e) => onChange?.(e, Number(e.target.value))}
      min={min}
      max={max}
      data-testid={props["data-testid"] || "slider"}
      {...props}
    />
  ),
}));

describe("AudioStageNode", () => {
  const mockWebSocket = {
    send: vi.fn(),
    close: vi.fn(),
    readyState: 1, // WebSocket.OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  } as any as WebSocket;

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
      voiceProtection: true,
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

  describe("Basic Rendering", () => {
    it("should render node with label and description", () => {
      renderNode();
      expect(screen.getByText("Test Node")).toBeInTheDocument();
      expect(screen.getByText("Test Description")).toBeInTheDocument();
    });

    it("should show enabled status chip", () => {
      renderNode();
      expect(screen.getByText("ENABLED")).toBeInTheDocument();
    });

    it("should show disabled status when disabled", () => {
      renderNode({
        data: { ...defaultNodeData, enabled: false },
      });
      expect(screen.getByText("DISABLED")).toBeInTheDocument();
    });

    it("should render metrics when provided", () => {
      renderNode();
      expect(screen.getByText(/12\.5ms/)).toBeInTheDocument();
      expect(screen.getByText(/\+2\.5dB/)).toBeInTheDocument();
      expect(screen.getByText(/15%/)).toBeInTheDocument();
    });
  });

  describe("Parameter Changes", () => {
    it("should call onParameterChange when slider is moved", async () => {
      const user = userEvent.setup({ delay: null });
      const onParameterChange = vi.fn();
      renderNode({ onParameterChange });

      // Find parameter sliders
      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2]; // After gain sliders

      // Change slider value
      await user.type(strengthSlider, "{ArrowRight}");

      // Should call immediately (local update)
      expect(onParameterChange).toHaveBeenCalled();
    });

    it("should debounce WebSocket parameter updates", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2];

      // Change parameter multiple times rapidly
      await user.type(strengthSlider, "{ArrowRight}{ArrowRight}{ArrowRight}");

      // WebSocket should not be called immediately
      expect(mockWebSocket.send).not.toHaveBeenCalled();

      // Fast-forward debounce timer (300ms)
      vi.advanceTimersByTime(300);

      // Now WebSocket should be called once
      await waitFor(() => {
        expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
      });

      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
      expect(sentMessage.type).toBe("update_stage");
      expect(sentMessage.stage_id).toBe("node-1");
      expect(sentMessage.parameters).toBeDefined();
    });

    it("should not send WebSocket messages when not active", async () => {
      const user = userEvent.setup({ delay: null });
      renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: false, // Not active
      });

      const sliders = screen.getAllByRole("slider");
      const strengthSlider = sliders[2];

      await user.type(strengthSlider, "{ArrowRight}");
      vi.advanceTimersByTime(300);

      expect(mockWebSocket.send).not.toHaveBeenCalled();
    });

    it("should show syncing indicator during parameter update", async () => {
      renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await userEvent.type(sliders[2], "{ArrowRight}");

      vi.advanceTimersByTime(300);

      // Should show syncing indicator briefly
      await waitFor(() => {
        expect(screen.getByTitle(/Syncing parameters/i)).toBeInTheDocument();
      });

      // After 500ms, syncing indicator should disappear
      vi.advanceTimersByTime(500);
      await waitFor(() => {
        expect(
          screen.queryByTitle(/Syncing parameters/i),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("Gain Controls", () => {
    it("should render input and output gain sliders", () => {
      renderNode();
      expect(screen.getByText("IN")).toBeInTheDocument();
      expect(screen.getByText("OUT")).toBeInTheDocument();
      expect(screen.getByText("+0.0dB")).toBeInTheDocument(); // Input gain
    });

    it("should call onGainChange for input gain", async () => {
      const onGainChange = vi.fn();
      renderNode({ onGainChange });

      const sliders = screen.getAllByRole("slider");
      const inputGainSlider = sliders[0];

      fireEvent.change(inputGainSlider, { target: { value: "5" } });

      expect(onGainChange).toHaveBeenCalledWith("node-1", "in", 5);
    });

    it("should call onGainChange for output gain", async () => {
      const onGainChange = vi.fn();
      renderNode({ onGainChange });

      const sliders = screen.getAllByRole("slider");
      const outputGainSlider = sliders[1];

      fireEvent.change(outputGainSlider, { target: { value: "-3" } });

      expect(onGainChange).toHaveBeenCalledWith("node-1", "out", -3);
    });

    it("should format gain values correctly", () => {
      renderNode({
        data: { ...defaultNodeData, gainIn: 12.5, gainOut: -6.2 },
      });

      expect(screen.getByText("+12.5dB")).toBeInTheDocument();
      expect(screen.getByText("-6.2dB")).toBeInTheDocument();
    });
  });

  describe("Enable/Disable Toggle", () => {
    it("should call onToggleEnabled when enabled chip is clicked", async () => {
      const onToggleEnabled = vi.fn();
      renderNode({ onToggleEnabled });

      const enabledChip = screen.getByText("ENABLED");
      await userEvent.click(enabledChip);

      expect(onToggleEnabled).toHaveBeenCalledWith("node-1", false);
    });

    it("should reduce opacity when disabled", () => {
      const { container } = renderNode({
        data: { ...defaultNodeData, enabled: false },
      });

      const card = container.querySelector('[class*="MuiCard"]');
      expect(card).toHaveStyle({ opacity: "0.6" });
    });
  });

  describe("Settings Panel", () => {
    it("should call onSettingsOpen when settings button is clicked", async () => {
      const onSettingsOpen = vi.fn();
      renderNode({ onSettingsOpen });

      const settingsButtons = screen.getAllByRole("button", {
        name: /settings/i,
      });
      await userEvent.click(settingsButtons[0]);

      expect(onSettingsOpen).toHaveBeenCalledWith("node-1");
    });
  });

  describe("Real-time Status Indicators", () => {
    it("should show sync icon when realtime is active", () => {
      renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: true,
      });

      expect(screen.getByTitle("Connected to backend")).toBeInTheDocument();
    });

    it("should show disconnect icon when not active", () => {
      renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: false,
      });

      expect(screen.getByTitle("Not connected to backend")).toBeInTheDocument();
    });

    it("should show green border when realtime active", () => {
      const { container } = renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: true,
      });

      const card = container.querySelector('[class*="MuiCard"]');
      const style = window.getComputedStyle(card!);
      expect(style.border).toContain("#4caf50");
    });
  });

  describe("Parameters Expansion", () => {
    it("should expand parameters section when clicked", async () => {
      renderNode();

      const parametersHeader = screen.getByText(/Parameters \(2\)/);
      await userEvent.click(parametersHeader);

      // Should show parameter sliders
      await waitFor(() => {
        expect(screen.getByText("strength:")).toBeInTheDocument();
        expect(screen.getByText("threshold:")).toBeInTheDocument();
      });
    });

    it("should collapse parameters when clicked again", async () => {
      renderNode();

      const parametersHeader = screen.getByText(/Parameters \(2\)/);

      // Expand
      await userEvent.click(parametersHeader);
      await waitFor(() => {
        expect(screen.getByText("strength:")).toBeInTheDocument();
      });

      // Collapse
      await userEvent.click(parametersHeader);
      await waitFor(() => {
        expect(screen.queryByText("strength:")).not.toBeInTheDocument();
      });
    });
  });

  describe("Event Propagation", () => {
    it("should stop propagation on slider container to prevent node dragging", () => {
      renderNode();

      const sliders = screen.getAllByRole("slider");
      const sliderContainer = sliders[0].parentElement?.parentElement;

      const mouseDownEvent = new MouseEvent("mousedown", {
        bubbles: true,
        cancelable: true,
      });

      const stopPropagation = vi.spyOn(mouseDownEvent, "stopPropagation");
      sliderContainer?.dispatchEvent(mouseDownEvent);

      expect(stopPropagation).toHaveBeenCalled();
    });
  });

  describe("WebSocket Connection States", () => {
    it("should not send messages when WebSocket is closed", async () => {
      const closedWebSocket = { ...mockWebSocket, readyState: 3 }; // CLOSED
      renderNode({
        websocket: closedWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await userEvent.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      expect(mockWebSocket.send).not.toHaveBeenCalled();
    });

    it("should handle WebSocket send errors gracefully", async () => {
      const errorWebSocket = {
        ...mockWebSocket,
        send: vi.fn().mockImplementation(() => {
          throw new Error("WebSocket send failed");
        }),
      };

      const consoleError = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      renderNode({
        websocket: errorWebSocket as any,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await userEvent.type(sliders[2], "{ArrowRight}");
      vi.advanceTimersByTime(300);

      await waitFor(() => {
        expect(consoleError).toHaveBeenCalledWith(
          "Failed to send parameter update:",
          expect.any(Error),
        );
      });

      consoleError.mockRestore();
    });
  });

  describe("Cleanup", () => {
    it("should clear debounce timers on unmount", async () => {
      const { unmount } = renderNode({
        websocket: mockWebSocket,
        isRealtimeActive: true,
      });

      const sliders = screen.getAllByRole("slider");
      await userEvent.type(sliders[2], "{ArrowRight}");

      // Unmount before debounce timer fires
      unmount();
      vi.advanceTimersByTime(300);

      // Should not send message after unmount
      expect(mockWebSocket.send).not.toHaveBeenCalled();
    });
  });
});
