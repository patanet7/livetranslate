import React, { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  NodeTypes,
  EdgeTypes,
  ReactFlowProvider,
  MarkerType,
  Panel,
  useKeyPress,
  useOnSelectionChange,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Box,
  Card,
  IconButton,
  Typography,
  Chip,
  Tooltip,
  Alert,
  Snackbar,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  useTheme,
} from '@mui/material';
import {
  Save,
  PlayArrow,
  Stop,
  Refresh,
  Delete,
  ContentCopy,
  ContentPaste,
  Undo,
  Redo,
  ZoomIn,
  ZoomOut,
  FitScreen,
  Timeline,
  Warning,
  CheckCircle,
  Error,
  Settings,
  ExpandMore,
  ExpandLess,
} from '@mui/icons-material';

import AudioStageNode from './AudioStageNode';
import { AudioComponent } from './ComponentLibrary';
import SettingsPanel from './SettingsPanel';

interface PipelineCanvasProps {
  initialPipeline?: PipelineData;
  onPipelineChange?: (pipeline: PipelineData) => void;
  onProcessingStart?: () => void;
  onProcessingStop?: () => void;
  isProcessing?: boolean;
  showMinimap?: boolean;
  showGrid?: boolean;
  height?: number;
}

interface PipelineData {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
  created: Date;
  modified: Date;
  metadata: {
    totalLatency: number;
    complexity: 'simple' | 'moderate' | 'complex';
    validated: boolean;
    errors: string[];
    warnings: string[];
  };
}

interface PipelineValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

// Custom node types
const nodeTypes: NodeTypes = {
  audioStage: AudioStageNode as any,
};

// Custom edge styles
const edgeTypes: EdgeTypes = {
  audioConnection: ({ sourceX, sourceY, targetX, targetY, style = {} }) => {
    const path = `M ${sourceX} ${sourceY} C ${sourceX + 100} ${sourceY}, ${targetX - 100} ${targetY}, ${targetX} ${targetY}`;
    
    return (
      <g>
        <path
          d={path}
          fill="none"
          stroke={style.stroke || '#4caf50'}
          strokeWidth={style.strokeWidth || 2}
          className="animated"
        />
        <circle cx={targetX} cy={targetY} r={3} fill={style.stroke || '#4caf50'} />
      </g>
    );
  },
};

const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  initialPipeline,
  onPipelineChange,
  onProcessingStart,
  onProcessingStop,
  isProcessing = false,
  showMinimap = true,
  showGrid = true,
  height = 600,
}) => {
  const theme = useTheme();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  
  const [nodes, setNodes, onNodesChange] = useNodesState(initialPipeline?.nodes || []);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialPipeline?.edges || []);
  
  const [selectedNodes, setSelectedNodes] = useState<Node[]>([]);
  const [copiedNodes, setCopiedNodes] = useState<Node[]>([]);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId?: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedNodeForSettings, setSelectedNodeForSettings] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<PipelineValidationResult | null>(null);
  const [validationExpanded, setValidationExpanded] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'warning' | 'info' }>({
    open: false,
    message: '',
    severity: 'info',
  });
  
  const [history, setHistory] = useState<{ nodes: Node[]; edges: Edge[] }[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  
  // Keyboard shortcuts
  const deletePressed = useKeyPress('Delete');
  const ctrlZ = useKeyPress(['Control+z', 'Meta+z']);
  const ctrlY = useKeyPress(['Control+y', 'Meta+y']);
  const ctrlC = useKeyPress(['Control+c', 'Meta+c']);
  const ctrlV = useKeyPress(['Control+v', 'Meta+v']);
  const ctrlS = useKeyPress(['Control+s', 'Meta+s']);

  // Track node selection
  useOnSelectionChange({
    onChange: ({ nodes }) => {
      console.log('Selection changed:', nodes.length, 'nodes selected');
      setSelectedNodes(nodes);
    },
  });

  // Validate pipeline whenever nodes or edges change
  useEffect(() => {
    validatePipeline();
  }, [nodes, edges]);

  // Save to history for undo/redo
  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push({ nodes: [...nodes], edges: [...edges] });
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  }, [nodes.length, edges.length]);

  // Keyboard shortcuts handlers
  useEffect(() => {
    if (deletePressed && selectedNodes.length > 0) {
      handleDeleteSelected();
    }
  }, [deletePressed]);

  useEffect(() => {
    if (ctrlZ) {
      handleUndo();
    }
  }, [ctrlZ]);

  useEffect(() => {
    if (ctrlY) {
      handleRedo();
    }
  }, [ctrlY]);

  useEffect(() => {
    if (ctrlC && selectedNodes.length > 0) {
      handleCopy();
    }
  }, [ctrlC]);

  useEffect(() => {
    if (ctrlV && copiedNodes.length > 0) {
      handlePaste();
    }
  }, [ctrlV]);

  useEffect(() => {
    if (ctrlS) {
      handleSave();
    }
  }, [ctrlS]);

  const validatePipeline = useCallback(() => {
    const errors: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    // Check for input node
    const inputNodes = nodes.filter(n => n.data.stageType === 'input');
    if (inputNodes.length === 0) {
      errors.push('Pipeline must have at least one input component');
    } else if (inputNodes.length > 1) {
      warnings.push('Multiple input components detected. Consider using a mixer if needed.');
    }

    // Check for output node
    const outputNodes = nodes.filter(n => n.data.stageType === 'output');
    if (outputNodes.length === 0) {
      errors.push('Pipeline must have at least one output component');
    }

    // Check for disconnected nodes
    const connectedNodeIds = new Set<string>();
    edges.forEach(edge => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });
    
    const disconnectedNodes = nodes.filter(n => 
      n.data.stageType !== 'input' && 
      n.data.stageType !== 'output' && 
      !connectedNodeIds.has(n.id)
    );
    
    if (disconnectedNodes.length > 0) {
      warnings.push(`${disconnectedNodes.length} component(s) are not connected to the pipeline`);
    }

    // Check for cycles
    if (hasCycle(nodes, edges)) {
      errors.push('Pipeline contains a cycle. Audio pipelines must be acyclic.');
    }

    // Calculate total latency
    const totalLatency = nodes.reduce((sum, node) => {
      if (node.data.metrics?.processingTimeMs) {
        return sum + node.data.metrics.processingTimeMs;
      }
      return sum;
    }, 0);

    if (totalLatency > 100) {
      warnings.push(`High total latency: ${totalLatency.toFixed(1)}ms. Consider optimizing for real-time performance.`);
    }

    // Suggestions
    const hasNoiseReduction = nodes.some(n => n.data.label.includes('Noise Reduction'));
    if (!hasNoiseReduction) {
      suggestions.push('Consider adding Noise Reduction for cleaner audio');
    }

    const hasNormalization = nodes.some(n => n.data.label.includes('LUFS Normalization'));
    if (!hasNormalization && outputNodes.some(n => n.data.label.includes('File Output'))) {
      suggestions.push('Consider adding LUFS Normalization for broadcast-compliant output');
    }

    setValidationResult({
      valid: errors.length === 0,
      errors,
      warnings,
      suggestions,
    });
  }, [nodes, edges]);

  const hasCycle = (nodes: Node[], edges: Edge[]): boolean => {
    const adjList = new Map<string, string[]>();
    nodes.forEach(node => adjList.set(node.id, []));
    edges.forEach(edge => {
      adjList.get(edge.source)?.push(edge.target);
    });

    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycleDFS = (nodeId: string): boolean => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      const neighbors = adjList.get(nodeId) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (hasCycleDFS(neighbor)) return true;
        } else if (recursionStack.has(neighbor)) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      return false;
    };

    for (const nodeId of adjList.keys()) {
      if (!visited.has(nodeId)) {
        if (hasCycleDFS(nodeId)) return true;
      }
    }

    return false;
  };

  const onConnect = useCallback((params: Connection) => {
    // Validate connection
    const sourceNode = nodes.find(n => n.id === params.source);
    const targetNode = nodes.find(n => n.id === params.target);

    if (!sourceNode || !targetNode) return;

    // Prevent connecting input to input or output to output
    if (sourceNode.data.stageType === 'input' && targetNode.data.stageType === 'input') {
      setSnackbar({ open: true, message: 'Cannot connect input to input', severity: 'error' });
      return;
    }
    if (sourceNode.data.stageType === 'output' && targetNode.data.stageType === 'output') {
      setSnackbar({ open: true, message: 'Cannot connect output to output', severity: 'error' });
      return;
    }

    // Create the edge
    setEdges((eds) => addEdge({
      ...params,
      type: 'audioConnection',
      animated: true,
      style: { stroke: '#4caf50', strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#4caf50',
      },
    }, eds));
  }, [nodes, setEdges]);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();

    const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
    if (!reactFlowBounds || !reactFlowInstance) return;

    const componentData = event.dataTransfer.getData('application/audioComponent');
    if (!componentData) return;

    const component: AudioComponent = JSON.parse(componentData);
    const position = reactFlowInstance.screenToFlowPosition({
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top,
    });

    const newNode: Node = {
      id: `${component.id}_${Date.now()}`,
      type: 'audioStage',
      position,
      data: {
        label: component.label,
        description: component.description,
        stageType: component.type,
        icon: component.icon,
        enabled: true,
        gainIn: 0,
        gainOut: 0,
        stageConfig: component.defaultConfig,
        metrics: {
          processingTimeMs: component.processingTime.target,
          targetLatencyMs: component.processingTime.max,
          qualityImprovement: 0,
          inputLevel: -20,
          outputLevel: -18,
          cpuUsage: 5,
        },
        isProcessing: false,
        status: 'idle',
        parameters: component.parameters.map(param => ({
          name: param.name,
          value: param.defaultValue,
          min: param.min || 0,
          max: param.max || 100,
          step: param.step || 1,
          unit: param.unit || '',
          description: param.description,
        })),
        onSettingsOpen: handleNodeSettingsOpen,
        onGainChange: handleGainChange,
        onParameterChange: handleParameterChange,
        onToggleEnabled: handleToggleEnabled,
      },
    };

    setNodes((nds) => nds.concat(newNode));
    setSnackbar({ open: true, message: `Added ${component.label} to pipeline`, severity: 'success' });
  }, [reactFlowInstance, setNodes]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleNodeSettingsOpen = (nodeId: string) => {
    setSelectedNodeForSettings(nodeId);
    setSettingsOpen(true);
  };

  const handleGainChange = (nodeId: string, type: 'in' | 'out', value: number) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              [type === 'in' ? 'gainIn' : 'gainOut']: value,
            },
          };
        }
        return node;
      })
    );
  };

  const handleParameterChange = (nodeId: string, paramName: string, value: number) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          const updatedParameters = node.data.parameters.map((param: any) => {
            if (param.name === paramName) {
              return { ...param, value };
            }
            return param;
          });
          return {
            ...node,
            data: {
              ...node.data,
              parameters: updatedParameters,
            },
          };
        }
        return node;
      })
    );
  };

  const handleToggleEnabled = (nodeId: string, enabled: boolean) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: {
              ...node.data,
              enabled,
            },
          };
        }
        return node;
      })
    );
  };

  const handleContextMenu = (event: React.MouseEvent, nodeId?: string) => {
    event.preventDefault();
    setContextMenu({
      x: event.clientX,
      y: event.clientY,
      nodeId,
    });
  };

  const handleDeleteSelected = () => {
    const selectedIds = selectedNodes.map(n => n.id);
    console.log('Deleting nodes:', selectedIds, 'Selected nodes count:', selectedNodes.length);
    if (selectedIds.length === 0) {
      setSnackbar({ open: true, message: 'No components selected for deletion', severity: 'warning' });
      return;
    }
    setNodes((nds) => nds.filter((node) => !selectedIds.includes(node.id)));
    setEdges((eds) => eds.filter((edge) => !selectedIds.includes(edge.source) && !selectedIds.includes(edge.target)));
    setSnackbar({ open: true, message: `Deleted ${selectedIds.length} component(s)`, severity: 'info' });
  };

  const handleCopy = () => {
    setCopiedNodes([...selectedNodes]);
    setSnackbar({ open: true, message: `Copied ${selectedNodes.length} component(s)`, severity: 'info' });
  };

  const handlePaste = () => {
    const pastedNodes: Node[] = copiedNodes.map((node) => ({
      ...node,
      id: `${node.id}_copy_${Date.now()}`,
      position: {
        x: node.position.x + 50,
        y: node.position.y + 50,
      },
    }));
    setNodes((nds) => nds.concat(pastedNodes));
    setSnackbar({ open: true, message: `Pasted ${pastedNodes.length} component(s)`, severity: 'success' });
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const previousState = history[historyIndex - 1];
      setNodes(previousState.nodes);
      setEdges(previousState.edges);
      setHistoryIndex(historyIndex - 1);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      setNodes(nextState.nodes);
      setEdges(nextState.edges);
      setHistoryIndex(historyIndex + 1);
    }
  };

  const handleSave = () => {
    const pipelineData: PipelineData = {
      id: initialPipeline?.id || `pipeline_${Date.now()}`,
      name: initialPipeline?.name || 'Untitled Pipeline',
      description: initialPipeline?.description || '',
      nodes,
      edges,
      created: initialPipeline?.created || new Date(),
      modified: new Date(),
      metadata: {
        totalLatency: nodes.reduce((sum, node) => sum + (node.data.metrics?.processingTimeMs || 0), 0),
        complexity: nodes.length > 10 ? 'complex' : nodes.length > 5 ? 'moderate' : 'simple',
        validated: validationResult?.valid || false,
        errors: validationResult?.errors || [],
        warnings: validationResult?.warnings || [],
      },
    };
    
    onPipelineChange?.(pipelineData);
    setSnackbar({ open: true, message: 'Pipeline saved successfully', severity: 'success' });
  };

  const handleClear = () => {
    setNodes([]);
    setEdges([]);
    setSnackbar({ open: true, message: 'Pipeline cleared', severity: 'info' });
  };

  const handleProcessingToggle = () => {
    if (isProcessing) {
      onProcessingStop?.();
    } else {
      if (validationResult?.valid) {
        onProcessingStart?.();
      } else {
        setSnackbar({ open: true, message: 'Fix validation errors before processing', severity: 'error' });
      }
    }
  };

  return (
    <Box 
      className="pipeline-studio"
      sx={{ 
        height, 
        width: '100%',
        position: 'absolute', 
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden',
        margin: 0,
        padding: 0
      }}
    >
      <ReactFlowProvider>
        <div 
          ref={reactFlowWrapper} 
          style={{ 
            width: '100%', 
            height: '100%', 
            margin: 0, 
            padding: 0,
            position: 'absolute',
            top: 0,
            left: 0
          }}
        >
          <ReactFlow
            style={{ 
              margin: 0, 
              padding: 0,
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%'
            }}
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeContextMenu={(event, node) => handleContextMenu(event, node.id)}
            onPaneContextMenu={handleContextMenu}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            nodesConnectable={true}
            nodesDraggable={true}
            elementsSelectable={true}
            selectNodesOnDrag={false}
            multiSelectionKeyCode={['Control', 'Meta']}
            deleteKeyCode={['Delete', 'Backspace']}
          >
            {/* Canvas Controls */}
            <Panel position="top-left">
              <Card sx={{ p: 1 }}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Tooltip title="Process Audio">
                    <IconButton
                      onClick={handleProcessingToggle}
                      color={isProcessing ? 'error' : 'success'}
                      disabled={!validationResult?.valid && !isProcessing}
                    >
                      {isProcessing ? <Stop /> : <PlayArrow />}
                    </IconButton>
                  </Tooltip>

                  <Divider orientation="vertical" flexItem />

                  <Tooltip title="Save Pipeline (Ctrl+S)">
                    <IconButton onClick={handleSave}>
                      <Save />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Clear Pipeline">
                    <IconButton onClick={handleClear}>
                      <Refresh />
                    </IconButton>
                  </Tooltip>

                  <Divider orientation="vertical" flexItem />

                  <Tooltip title="Undo (Ctrl+Z)">
                    <IconButton onClick={handleUndo} disabled={historyIndex <= 0}>
                      <Undo />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Redo (Ctrl+Y)">
                    <IconButton onClick={handleRedo} disabled={historyIndex >= history.length - 1}>
                      <Redo />
                    </IconButton>
                  </Tooltip>

                  <Divider orientation="vertical" flexItem />

                  <Tooltip title="Zoom In">
                    <IconButton onClick={() => reactFlowInstance?.zoomIn()}>
                      <ZoomIn />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Zoom Out">
                    <IconButton onClick={() => reactFlowInstance?.zoomOut()}>
                      <ZoomOut />
                    </IconButton>
                  </Tooltip>

                  <Tooltip title="Fit View">
                    <IconButton onClick={() => reactFlowInstance?.fitView()}>
                      <FitScreen />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Card>
            </Panel>

            {/* Pipeline Status */}
            <Panel position="top-center">
              <Box>
                <Card sx={{ p: 1, minWidth: 300 }}>
                  <Box display="flex" alignItems="center" justifyContent="center" gap={2}>
                    <Typography variant="body2" fontWeight="bold">
                      Pipeline Status:
                    </Typography>
                    {validationResult?.valid ? (
                      <Chip
                        icon={<CheckCircle />}
                        label="Valid"
                        color="success"
                        size="small"
                      />
                    ) : (
                      <Chip
                        icon={<Error />}
                        label={`${validationResult?.errors.length || 0} Errors`}
                        color="error"
                        size="small"
                        onClick={() => setValidationExpanded(!validationExpanded)}
                        onDelete={() => setValidationExpanded(!validationExpanded)}
                        deleteIcon={validationExpanded ? <ExpandLess /> : <ExpandMore />}
                        sx={{ 
                          cursor: 'pointer',
                          '& .MuiChip-deleteIcon': {
                            color: 'inherit',
                            '&:hover': {
                              color: 'inherit'
                            }
                          }
                        }}
                      />
                    )}
                    {validationResult && validationResult.warnings.length > 0 && (
                      <Chip
                        icon={<Warning />}
                        label={`${validationResult.warnings.length} Warnings`}
                        color="warning"
                        size="small"
                        onClick={() => setValidationExpanded(!validationExpanded)}
                        onDelete={() => setValidationExpanded(!validationExpanded)}
                        deleteIcon={validationExpanded ? <ExpandLess /> : <ExpandMore />}
                        sx={{ 
                          cursor: 'pointer',
                          '& .MuiChip-deleteIcon': {
                            color: 'inherit',
                            '&:hover': {
                              color: 'inherit'
                            }
                          }
                        }}
                      />
                    )}
                    <Chip
                      icon={<Timeline />}
                      label={`${nodes.reduce((sum, node) => sum + (node.data.metrics?.processingTimeMs || 0), 0).toFixed(1)}ms`}
                      size="small"
                    />
                  </Box>
                </Card>
                
                {/* Expandable Validation Details */}
                {validationExpanded && validationResult && (validationResult.errors.length > 0 || validationResult.warnings.length > 0) && (
                  <Card sx={{ mt: 1, p: 2, minWidth: 400, maxWidth: 600 }}>
                    <Box>
                      {validationResult.errors.length > 0 && (
                        <Box mb={validationResult.warnings.length > 0 ? 2 : 0}>
                          <Typography variant="subtitle2" color="error.main" fontWeight="bold" mb={1}>
                            Errors ({validationResult.errors.length})
                          </Typography>
                          {validationResult.errors.map((error, index) => (
                            <Alert key={index} severity="error" sx={{ mb: 1 }}>
                              {error}
                            </Alert>
                          ))}
                        </Box>
                      )}
                      
                      {validationResult.warnings.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" color="warning.main" fontWeight="bold" mb={1}>
                            Warnings ({validationResult.warnings.length})
                          </Typography>
                          {validationResult.warnings.map((warning, index) => (
                            <Alert key={index} severity="warning" sx={{ mb: 1 }}>
                              {warning}
                            </Alert>
                          ))}
                        </Box>
                      )}
                      
                      {validationResult.suggestions.length > 0 && (
                        <Box mt={2}>
                          <Typography variant="subtitle2" color="info.main" fontWeight="bold" mb={1}>
                            Suggestions ({validationResult.suggestions.length})
                          </Typography>
                          {validationResult.suggestions.map((suggestion, index) => (
                            <Alert key={index} severity="info" sx={{ mb: 1 }}>
                              {suggestion}
                            </Alert>
                          ))}
                        </Box>
                      )}
                    </Box>
                  </Card>
                )}
              </Box>
            </Panel>

            {/* Pipeline Info */}
            <Panel position="top-right">
              <Card sx={{ p: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {nodes.length} components, {edges.length} connections
                </Typography>
              </Card>
            </Panel>


            <Controls />
            {showMinimap && (
              <MiniMap
                position="top-right"
                style={{
                  backgroundColor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.8)' 
                    : 'rgba(255, 255, 255, 0.8)',
                  border: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.12)' 
                    : 'rgba(0, 0, 0, 0.12)'}`,
                  borderRadius: '8px',
                }}
                nodeStrokeColor={(node) => {
                  switch (node.data.stageType) {
                    case 'input': return theme.palette.primary.main;
                    case 'processing': return theme.palette.success.main;
                    case 'output': return theme.palette.warning.main;
                    default: return theme.palette.grey[500];
                  }
                }}
                nodeColor={(node) => {
                  const alpha = theme.palette.mode === 'dark' ? 0.3 : 0.2;
                  switch (node.data.stageType) {
                    case 'input': return `${theme.palette.primary.main}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
                    case 'processing': return `${theme.palette.success.main}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
                    case 'output': return `${theme.palette.warning.main}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
                    default: return `${theme.palette.grey[500]}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
                  }
                }}
                nodeBorderRadius={8}
                maskColor={theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.1)' 
                  : 'rgba(0, 0, 0, 0.1)'}
              />
            )}
            {showGrid && <Background variant={BackgroundVariant.Dots} gap={16} size={1} style={{ margin: 0, padding: 0 }} />}
          </ReactFlow>
        </div>

        {/* Context Menu */}
        <Menu
          open={contextMenu !== null}
          onClose={() => setContextMenu(null)}
          anchorReference="anchorPosition"
          anchorPosition={
            contextMenu !== null
              ? { top: contextMenu.y, left: contextMenu.x }
              : undefined
          }
        >
          {contextMenu?.nodeId && (
            <>
              <MenuItem onClick={() => {
                handleCopy();
                setContextMenu(null);
              }}>
                <ListItemIcon><ContentCopy fontSize="small" /></ListItemIcon>
                <ListItemText>Copy</ListItemText>
              </MenuItem>
              <MenuItem onClick={() => {
                handleDeleteSelected();
                setContextMenu(null);
              }}>
                <ListItemIcon><Delete fontSize="small" /></ListItemIcon>
                <ListItemText>Delete</ListItemText>
              </MenuItem>
              <MenuItem onClick={() => {
                handleNodeSettingsOpen(contextMenu.nodeId!);
                setContextMenu(null);
              }}>
                <ListItemIcon><Settings fontSize="small" /></ListItemIcon>
                <ListItemText>Settings</ListItemText>
              </MenuItem>
              <Divider />
            </>
          )}
          <MenuItem onClick={() => {
            handlePaste();
            setContextMenu(null);
          }} disabled={copiedNodes.length === 0}>
            <ListItemIcon><ContentPaste fontSize="small" /></ListItemIcon>
            <ListItemText>Paste</ListItemText>
          </MenuItem>
        </Menu>

        {/* Settings Dialog */}
        {selectedNodeForSettings && (
          <SettingsPanel
            open={settingsOpen}
            onClose={() => {
              setSettingsOpen(false);
              setSelectedNodeForSettings(null);
            }}
            nodeId={selectedNodeForSettings}
            nodeData={nodes.find(n => n.id === selectedNodeForSettings)?.data}
            onParameterChange={handleParameterChange}
            onGainChange={handleGainChange}
            onToggleEnabled={handleToggleEnabled}
          />
        )}

        {/* Snackbar */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={3000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert
            onClose={() => setSnackbar({ ...snackbar, open: false })}
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </ReactFlowProvider>
    </Box>
  );
};

export default PipelineCanvas;
export type { PipelineData, PipelineValidationResult };