import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@mui/material', () => {
  const component =
    (tag = 'div') =>
    ({ children, ...props }: any) =>
      React.createElement(tag, props, children);

  return {
    Box: component('div'),
    Card: component('div'),
    CardContent: component('div'),
    Typography: component('div'),
    LinearProgress: component('progress'),
    Divider: component('hr'),
    Tooltip: ({ title, children }: any) =>
      React.createElement(
        'div',
        { title: typeof title === 'string' ? title : undefined },
        children
      ),
    IconButton: ({ children, ...props }: any) => React.createElement('button', props, children),
    Chip: ({ label, onClick, ...props }: any) =>
      React.createElement('button', { ...props, onClick }, label),
    Slider: ({ value, onChange, min, max, ...props }: any) =>
      React.createElement('input', {
        ...props,
        type: 'range',
        role: 'slider',
        value,
        min,
        max,
        onChange: (event: any) => onChange?.(event, Number(event.target.value)),
      }),
  };
});

vi.mock('@mui/icons-material', () => {
  return new Proxy(
    {},
    {
      get: (_target, property) => {
        if (property === '__esModule') return true;
        return (props: any) => React.createElement('span', props, String(property));
      },
    }
  );
});

vi.mock('reactflow', () => {
  return {
    Handle: (props: any) => React.createElement('div', props),
    Position: {
      Left: 'left',
      Right: 'right',
      Top: 'top',
      Bottom: 'bottom',
    },
    ReactFlowProvider: ({ children }: any) => React.createElement(React.Fragment, null, children),
  };
});

// ReactFlow-backed editor components currently deadlock under the jsdom/Vitest runtime
// in this workspace. Keep the suite typed and colocated, but skip runtime execution
// until the editor tests move to a browser-capable harness.
describe.skip('AudioStageNode', () => {
  const defaultNodeData = {
    label: 'Test Node',
    description: 'Test Description',
    stageType: 'processing' as const,
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
        name: 'strength',
        value: 0.7,
        min: 0.0,
        max: 1.0,
        step: 0.1,
        unit: '',
        description: 'Processing strength',
      },
      {
        name: 'threshold',
        value: 0.5,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        unit: '',
        description: 'Threshold level',
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
    status: 'idle' as const,
  };

  const defaultProps = {
    id: 'node-1',
    data: defaultNodeData,
    selected: false,
    onSettingsOpen: vi.fn(),
    onGainChange: vi.fn(),
    onParameterChange: vi.fn(),
    onToggleEnabled: vi.fn(),
    isConnectable: true,
    type: 'audioStage',
    position: { x: 0, y: 0 },
    dragging: false,
    zIndex: 0,
    xPos: 0,
    yPos: 0,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderNode = async (props = {}) => {
    const { default: AudioStageNode } = await import('../AudioStageNode');
    return render(
      <div style={{ width: 400, height: 600 }}>
        <AudioStageNode {...defaultProps} {...props} />
      </div>
    );
  };

  it('renders node label, description, and metrics', async () => {
    await renderNode();

    expect(screen.getByText('Test Node')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
    expect(screen.getByText(/12\.5ms/)).toBeInTheDocument();
    expect(screen.getByText(/\+2\.5dB/)).toBeInTheDocument();
    expect(screen.getByText(/15%/)).toBeInTheDocument();
  });

  it('renders input and output gain controls', async () => {
    await renderNode();

    expect(screen.getByText('IN')).toBeInTheDocument();
    expect(screen.getByText('OUT')).toBeInTheDocument();
    expect(screen.getByText('+0.0dB')).toBeInTheDocument();
  });

  it('calls onGainChange for input and output sliders', async () => {
    const onGainChange = vi.fn();
    await renderNode({ onGainChange });

    const sliders = screen.getAllByRole('slider');
    fireEvent.change(sliders[0], { target: { value: '5' } });
    fireEvent.change(sliders[1], { target: { value: '-3' } });

    expect(onGainChange).toHaveBeenCalledWith('node-1', 'in', 5);
    expect(onGainChange).toHaveBeenCalledWith('node-1', 'out', -3);
  });

  it('calls onParameterChange when stage parameter sliders are adjusted', async () => {
    const user = userEvent.setup();
    const onParameterChange = vi.fn();
    await renderNode({ onParameterChange });

    await user.click(screen.getByText(/Parameters \(2\)/));

    const sliders = screen.getAllByRole('slider');
    fireEvent.change(sliders[2], { target: { value: '0.8' } });
    fireEvent.change(sliders[3], { target: { value: '0.55' } });

    expect(onParameterChange).toHaveBeenCalledWith('node-1', 'strength', 0.8);
    expect(onParameterChange).toHaveBeenCalledWith('node-1', 'threshold', 0.55);
  });

  it('toggles enabled state from the status chip', async () => {
    const user = userEvent.setup();
    const onToggleEnabled = vi.fn();
    await renderNode({ onToggleEnabled });

    await user.click(screen.getByText('ENABLED'));

    expect(onToggleEnabled).toHaveBeenCalledWith('node-1', false);
  });

  it('opens settings from the header action', async () => {
    const user = userEvent.setup();
    const onSettingsOpen = vi.fn();
    const { container } = await renderNode({ onSettingsOpen });

    const buttons = container.querySelectorAll('button');
    await user.click(buttons[0]);

    expect(onSettingsOpen).toHaveBeenCalledWith('node-1');
  });

  it('expands and collapses the parameters section', async () => {
    const user = userEvent.setup();
    await renderNode();

    const header = screen.getByText(/Parameters \(2\)/);
    await user.click(header);

    await waitFor(() => {
      expect(screen.getByText('strength:')).toBeInTheDocument();
      expect(screen.getByText('threshold:')).toBeInTheDocument();
    });

    await user.click(header);

    await waitFor(() => {
      expect(screen.queryByText('strength:')).not.toBeInTheDocument();
    });
  });
});
