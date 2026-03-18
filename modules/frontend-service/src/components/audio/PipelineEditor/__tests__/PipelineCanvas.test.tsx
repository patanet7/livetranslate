import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import type { PipelineData } from '../PipelineCanvas';

vi.mock('@mui/material', () => {
  const component =
    (tag = 'div') =>
    ({ children, ...props }: any) =>
      React.createElement(tag, props, children);

  return {
    Box: component('div'),
    Card: component('div'),
    IconButton: ({ children, ...props }: any) => React.createElement('button', props, children),
    Typography: component('div'),
    Chip: ({ label, ...props }: any) => React.createElement('div', props, label),
    Tooltip: ({ title, children }: any) =>
      React.createElement(
        'div',
        { title: typeof title === 'string' ? title : undefined },
        children
      ),
    Alert: component('div'),
    Snackbar: ({ open, children }: any) => (open ? React.createElement('div', {}, children) : null),
    Menu: ({ open, children }: any) => (open ? React.createElement('div', {}, children) : null),
    MenuItem: component('button'),
    ListItemIcon: component('span'),
    ListItemText: ({ primary }: any) => React.createElement('span', {}, primary),
    Divider: component('hr'),
    useTheme: () => ({
      palette: {
        primary: { main: '#1976d2' },
        success: { main: '#4caf50' },
        warning: { main: '#ff9800' },
        error: { main: '#f44336' },
        background: { paper: '#fff', default: '#fff' },
        divider: '#ccc',
      },
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
  const MockReactFlow = ({ children, nodes, edges }: any) => (
    <div data-testid="react-flow">
      <div data-testid="nodes-count">{nodes.length}</div>
      <div data-testid="edges-count">{edges.length}</div>
      {children}
    </div>
  );

  return {
    ReactFlowProvider: ({ children }: any) => React.createElement(React.Fragment, null, children),
    Background: () => null,
    BackgroundVariant: { Dots: 'dots' },
    Controls: () => null,
    MiniMap: () => null,
    Panel: ({ children }: any) => React.createElement('div', {}, children),
    MarkerType: { ArrowClosed: 'arrowclosed' },
    ConnectionLineType: { Bezier: 'bezier' },
    Handle: (props: any) => React.createElement('div', props),
    Position: {
      Left: 'left',
      Right: 'right',
      Top: 'top',
      Bottom: 'bottom',
    },
    addEdge: vi.fn((edge: any, edges: any[]) => [...edges, edge]),
    useNodesState: (initialNodes: any[]) => [initialNodes, vi.fn(), vi.fn()],
    useEdgesState: (initialEdges: any[]) => [initialEdges, vi.fn(), vi.fn()],
    useKeyPress: () => false,
    useOnSelectionChange: vi.fn(),
    default: MockReactFlow,
    ReactFlow: MockReactFlow,
  };
});

// ReactFlow-backed editor components currently deadlock under the jsdom/Vitest runtime
// in this workspace. Keep the suite typed and colocated, but skip runtime execution
// until the editor tests move to a browser-capable harness.
describe.skip('PipelineCanvas', () => {
  const defaultProps = {
    onPipelineChange: vi.fn(),
    onProcessingStart: vi.fn(),
    onProcessingStop: vi.fn(),
    isProcessing: false,
    showMinimap: true,
    showGrid: true,
    height: 600,
  };

  const buildPipeline = (overrides: Partial<PipelineData> = {}): PipelineData => ({
    id: 'test-pipeline',
    name: 'Test Pipeline',
    description: 'Test',
    nodes: [],
    edges: [],
    created: new Date(),
    modified: new Date(),
    metadata: {
      totalLatency: 0,
      complexity: 'simple',
      validated: false,
      errors: [],
      warnings: [],
    },
    ...overrides,
  });

  const buildNode = (id: string, stageType: 'input' | 'processing' | 'output') => ({
    id,
    type: 'audioStage',
    position: { x: 100, y: 100 },
    data: {
      label: id,
      description: `${id} description`,
      stageType,
      icon: () => null,
      enabled: true,
      gainIn: 0,
      gainOut: 0,
      stageConfig: {},
      parameters: [],
      isProcessing: false,
      status: 'idle' as const,
    },
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderPipelineCanvas = async (props = {}) => {
    const { default: PipelineCanvas } = await import('../PipelineCanvas');
    return render(
      <div style={{ width: '100%', height: '600px' }}>
        <PipelineCanvas {...defaultProps} {...props} />
      </div>
    );
  };

  it('starts with an empty pipeline by default', async () => {
    await renderPipelineCanvas();

    expect(screen.getByTestId('nodes-count')).toHaveTextContent('0');
    expect(screen.getByTestId('edges-count')).toHaveTextContent('0');
  });

  it('renders nodes and edges from the initial pipeline', async () => {
    const initialPipeline = buildPipeline({
      nodes: [buildNode('input-1', 'input'), buildNode('output-1', 'output')],
      edges: [
        {
          id: 'edge-1',
          source: 'input-1',
          target: 'output-1',
          type: 'audioConnection',
        },
      ],
    });

    await renderPipelineCanvas({ initialPipeline });

    await waitFor(() => {
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('2');
      expect(screen.getByTestId('edges-count')).toHaveTextContent('1');
    });
  });

  it('updates rendered counts when the pipeline changes', async () => {
    const initialPipeline = buildPipeline({
      nodes: [buildNode('input-1', 'input')],
    });

    const { rerender } = await renderPipelineCanvas({ initialPipeline });
    const { default: PipelineCanvas } = await import('../PipelineCanvas');

    rerender(
      <PipelineCanvas
        {...defaultProps}
        initialPipeline={buildPipeline({
          nodes: [buildNode('input-1', 'input'), buildNode('output-1', 'output')],
          edges: [
            {
              id: 'edge-1',
              source: 'input-1',
              target: 'output-1',
              type: 'audioConnection',
            },
          ],
        })}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('2');
      expect(screen.getByTestId('edges-count')).toHaveTextContent('1');
    });
  });

  it('renders safely when realtime mode is toggled', async () => {
    const initialPipeline = buildPipeline({
      nodes: [buildNode('input-1', 'input'), buildNode('output-1', 'output')],
    });

    const { rerender } = await renderPipelineCanvas({
      initialPipeline,
      isRealtimeActive: false,
    });
    const { default: PipelineCanvas } = await import('../PipelineCanvas');

    rerender(
      <PipelineCanvas
        {...defaultProps}
        initialPipeline={initialPipeline}
        isRealtimeActive={true}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('nodes-count')).toHaveTextContent('2');
    });
  });
});
