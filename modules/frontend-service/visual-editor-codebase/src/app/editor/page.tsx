"use client";

import React, { useState, useCallback, useRef, useEffect } from 'react';
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
  Handle,
  Position,
  NodeProps,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { 
  Plus, 
  Save, 
  Play, 
  RotateCcw, 
  Settings, 
  FileText, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Users,
  Shield,
  Eye,
  Database,
  GitBranch,
  Activity,
  Zap,
  Brain,
  Filter,
  Target,
  Bell
} from 'lucide-react';

// Compliance Rule Types based on the model
enum RuleStatus {
  DRAFT = "draft",
  TESTING = "testing",
  ACTIVE = "active",
  DEPRECATED = "deprecated"
}

enum RuleCategory {
  KYC = "kyc",
  AML = "aml",
  SANCTIONS = "sanctions",
  PEP = "pep",
  RISK_ASSESSMENT = "risk_assessment",
  COMPLIANCE = "compliance"
}

interface RuleCondition {
  field: string;
  operator: string;
  value: any;
  logic?: string;
}

interface RuleAction {
  action_type: string;
  severity?: string;
  message?: string;
  escalation_required: boolean;
}

interface ComplianceRule {
  rule_id: string;
  name: string;
  description: string;
  category: RuleCategory;
  status: RuleStatus;
  version: number;
  author: string;
  tags: string[];
  conditions: RuleCondition[];
  actions: RuleAction[];
  priority: number;
  is_enabled: boolean;
  created_at: Date;
  updated_at: Date;
}

// Node Types for Compliance Rules
const COMPLIANCE_NODE_TYPES = {
  CONDITION: 'condition',
  ACTION: 'action',
  LOGIC: 'logic',
  INPUT: 'input',
  OUTPUT: 'output'
};

const RULE_OPERATORS = [
  { value: 'equals', label: 'Equals' },
  { value: 'not_equals', label: 'Not Equals' },
  { value: 'contains', label: 'Contains' },
  { value: 'not_contains', label: 'Does Not Contain' },
  { value: 'greater_than', label: 'Greater Than' },
  { value: 'less_than', label: 'Less Than' },
  { value: 'greater_equal', label: 'Greater or Equal' },
  { value: 'less_equal', label: 'Less or Equal' },
  { value: 'in_list', label: 'In List' },
  { value: 'not_in_list', label: 'Not In List' },
  { value: 'regex', label: 'Regex Match' },
  { value: 'is_empty', label: 'Is Empty' },
  { value: 'is_not_empty', label: 'Is Not Empty' }
];

const ACTION_TYPES = [
  { value: 'flag', label: 'Flag', icon: '🚩', color: 'bg-yellow-500' },
  { value: 'block', label: 'Block', icon: '🛑', color: 'bg-red-500' },
  { value: 'approve', label: 'Approve', icon: '✅', color: 'bg-green-500' },
  { value: 'review', label: 'Manual Review', icon: '👁️', color: 'bg-blue-500' },
  { value: 'escalate', label: 'Escalate', icon: '⬆️', color: 'bg-orange-500' },
  { value: 'notify', label: 'Notify', icon: '🔔', color: 'bg-purple-500' },
  { value: 'log', label: 'Log Event', icon: '📝', color: 'bg-gray-500' },
  { value: 'quarantine', label: 'Quarantine', icon: '🔒', color: 'bg-red-600' }
];

const SEVERITY_LEVELS = [
  { value: 'low', label: 'Low', color: 'text-green-400' },
  { value: 'medium', label: 'Medium', color: 'text-yellow-400' },
  { value: 'high', label: 'High', color: 'text-orange-400' },
  { value: 'critical', label: 'Critical', color: 'text-red-400' }
];

// Component Library for Compliance Rules
const COMPLIANCE_COMPONENTS = {
  conditions: [
    { 
      type: 'customer_data', 
      label: 'Customer Data Check', 
      icon: Users, 
      description: 'Validate customer information',
      defaultCondition: { field: 'customer.age', operator: 'greater_than', value: 18 }
    },
    { 
      type: 'document_verification', 
      label: 'Document Verification', 
      icon: FileText, 
      description: 'Check document validity',
      defaultCondition: { field: 'document.type', operator: 'equals', value: 'passport' }
    },
    { 
      type: 'risk_score', 
      label: 'Risk Score Check', 
      icon: AlertTriangle, 
      description: 'Evaluate risk scoring',
      defaultCondition: { field: 'risk_score', operator: 'less_than', value: 0.7 }
    },
    { 
      type: 'sanctions_check', 
      label: 'Sanctions Screening', 
      icon: Shield, 
      description: 'Screen against sanctions lists',
      defaultCondition: { field: 'sanctions.match', operator: 'equals', value: false }
    },
    { 
      type: 'pep_check', 
      label: 'PEP Screening', 
      icon: Eye, 
      description: 'Politically Exposed Person check',
      defaultCondition: { field: 'pep.status', operator: 'equals', value: false }
    },
    { 
      type: 'transaction_amount', 
      label: 'Transaction Amount', 
      icon: Database, 
      description: 'Check transaction value',
      defaultCondition: { field: 'transaction.amount', operator: 'less_than', value: 10000 }
    }
  ],
  logic: [
    { type: 'and_gate', label: 'AND Logic', icon: GitBranch, description: 'All conditions must be true' },
    { type: 'or_gate', label: 'OR Logic', icon: GitBranch, description: 'Any condition must be true' },
    { type: 'not_gate', label: 'NOT Logic', icon: GitBranch, description: 'Negate condition result' }
  ],
  actions: [
    { type: 'approval_action', label: 'Auto Approval', icon: CheckCircle, description: 'Automatically approve', actionType: 'approve' },
    { type: 'flag_action', label: 'Flag Transaction', icon: AlertTriangle, description: 'Flag for review', actionType: 'flag' },
    { type: 'block_action', label: 'Block Transaction', icon: XCircle, description: 'Block transaction', actionType: 'block' },
    { type: 'notify_action', label: 'Send Notification', icon: Bell, description: 'Send alert notification', actionType: 'notify' },
    { type: 'escalate_action', label: 'Escalate Review', icon: Target, description: 'Escalate to supervisor', actionType: 'escalate' }
  ]
};

// Custom Node Component for Compliance Rules
const ComplianceNode: React.FC<NodeProps> = ({ data, selected, id }) => {
  const getNodeIcon = () => {
    switch (data.nodeType) {
      case 'condition':
        return <Filter className="w-4 h-4" />;
      case 'action':
        return <Zap className="w-4 h-4" />;
      case 'logic':
        return <Brain className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const getNodeColor = () => {
    switch (data.nodeType) {
      case 'condition':
        return 'from-blue-500/20 to-blue-600/20 border-blue-500/30';
      case 'action':
        return 'from-green-500/20 to-green-600/20 border-green-500/30';
      case 'logic':
        return 'from-purple-500/20 to-purple-600/20 border-purple-500/30';
      default:
        return 'from-gray-500/20 to-gray-600/20 border-gray-500/30';
    }
  };

  return (
    <div className={`
      relative p-3 rounded-lg border-2 min-w-[180px] bg-gradient-to-br transition-all duration-200
      ${getNodeColor()}
      ${selected ? 'shadow-lg border-accent-green' : 'hover:shadow-md'}
    `}>
      {/* Input Handle */}
      {data.nodeType !== 'input' && (
        <Handle
          type="target"
          position={Position.Left}
          className="w-3 h-3 bg-accent-green border-2 border-bg-card opacity-0 group-hover:opacity-100 transition-opacity"
        />
      )}
      
      {/* Node Content */}
      <div className="flex items-center gap-2 mb-1">
        <div className="p-1 rounded bg-black/20">
          {getNodeIcon()}
        </div>
        <span className="font-medium text-sm text-text-primary">{data.label}</span>
      </div>
      
      {data.description && (
        <p className="text-xs text-text-secondary leading-tight">{data.description}</p>
      )}

      {/* Condition/Action Details */}
      {data.condition && (
        <div className="mt-2 p-2 bg-black/20 rounded text-xs">
          <span className="text-accent-green">{data.condition.field}</span>
          <span className="text-text-secondary"> {data.condition.operator} </span>
          <span className="text-text-primary">{data.condition.value}</span>
        </div>
      )}

      {data.action && (
        <div className="mt-2 p-2 bg-black/20 rounded text-xs flex items-center gap-1">
          <Badge variant="secondary" className="text-xs">
            {data.action.action_type}
          </Badge>
          {data.action.severity && (
            <Badge variant="outline" className="text-xs">
              {data.action.severity}
            </Badge>
          )}
        </div>
      )}
      
      {/* Output Handle */}
      {data.nodeType !== 'action' && (
        <Handle
          type="source"
          position={Position.Right}
          className="w-3 h-3 bg-accent-green border-2 border-bg-card opacity-0 group-hover:opacity-100 transition-opacity"
        />
      )}
    </div>
  );
};

const nodeTypes = {
  complianceNode: ComplianceNode,
};

export default function ComplianceRulesEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [currentRule, setCurrentRule] = useState<Partial<ComplianceRule>>({
    name: '',
    description: '',
    category: RuleCategory.COMPLIANCE,
    status: RuleStatus.DRAFT,
    author: 'Current User',
    tags: [],
    conditions: [],
    actions: [],
    priority: 5,
    is_enabled: true,
    version: 1
  });
  const [stats, setStats] = useState({
    totalNodes: 0,
    conditions: 0,
    actions: 0,
    connections: 0
  });
  const [activity, setActivity] = useState<string[]>([]);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Add new activity log entry
  const addActivity = useCallback((message: string) => {
    setActivity(prev => [`${new Date().toLocaleTimeString()}: ${message}`, ...prev.slice(0, 9)]);
  }, []);

  // Update stats when nodes or edges change
  useEffect(() => {
    setStats({
      totalNodes: nodes.length,
      conditions: nodes.filter(n => n.data.nodeType === 'condition').length,
      actions: nodes.filter(n => n.data.nodeType === 'action').length,
      connections: edges.length
    });
  }, [nodes, edges]);

  // Handle connection between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      // Validate connection
      if (params.source === params.target) return;
      
      // Check if connection already exists
      const existingConnection = edges.find(
        edge => edge.source === params.source && edge.target === params.target
      );
      if (existingConnection) return;

      setEdges((eds) => addEdge({
        ...params,
        type: 'smoothstep',
        style: { stroke: 'var(--color-accent-green)', strokeWidth: 2 },
        animated: true
      }, eds));
      
      addActivity(`Connected ${params.source} to ${params.target}`);
    },
    [edges, addActivity]
  );

  // Handle drag over
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowInstance || !reactFlowWrapper.current) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const componentData = event.dataTransfer.getData('application/reactflow');
      
      if (!componentData) return;

      const component = JSON.parse(componentData);
      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: `${component.type}_${Date.now()}`,
        type: 'complianceNode',
        position,
        data: {
          label: component.label,
          description: component.description,
          nodeType: getNodeType(component.type),
          condition: component.defaultCondition,
          action: component.actionType ? { 
            action_type: component.actionType,
            severity: 'medium',
            escalation_required: false 
          } : null
        },
      };

      setNodes((nds) => nds.concat(newNode));
      setSelectedNode(newNode); // Select the newly added node
      addActivity(`Added ${component.label} component`);
    },
    [reactFlowInstance, addActivity]
  );

  const getNodeType = (componentType: string) => {
    if (componentType.includes('action')) return 'action';
    if (componentType.includes('gate')) return 'logic';
    return 'condition';
  };

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  // Handle pane click to deselect node
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // Drag start handler for component library
  const onDragStart = (event: React.DragEvent, component: any) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(component));
    event.dataTransfer.effectAllowed = 'move';
  };

  // Save rule function (mock API call)
  const saveRule = async () => {
    try {
      // Extract conditions and actions from nodes
      const extractedConditions = nodes
        .filter(n => n.data.condition)
        .map(n => n.data.condition);
      
      const extractedActions = nodes
        .filter(n => n.data.action)
        .map(n => n.data.action);

      const ruleToSave: ComplianceRule = {
        ...currentRule as ComplianceRule,
        rule_id: currentRule.rule_id || `rule_${Date.now()}`,
        conditions: extractedConditions,
        actions: extractedActions,
        created_at: new Date(),
        updated_at: new Date()
      };

      // Mock API call
      console.log('Saving rule:', ruleToSave);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      addActivity('Rule saved successfully');
      alert('Rule saved successfully!');
    } catch (error) {
      console.error('Error saving rule:', error);
      addActivity('Error saving rule');
    }
  };

  // Test rule function (mock)
  const testRule = async () => {
    addActivity('Running rule test...');
    
    // Simulate test
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const passed = Math.random() > 0.3;
    addActivity(`Test ${passed ? 'passed' : 'failed'} - ${Math.round(Math.random() * 1000)}ms`);
  };

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      {/* Header */}
      <header className="border-b border-border-subtle bg-bg-card">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-accent-green to-green-400 rounded-lg flex items-center justify-center">
                <Shield className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Compliance Rules Editor</h1>
                <p className="text-sm text-text-secondary">Visual business logic designer</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={testRule}>
              <Play className="w-4 h-4 mr-1" />
              Test Rule
            </Button>
            <Button onClick={saveRule} size="sm">
              <Save className="w-4 h-4 mr-1" />
              Save Rule
            </Button>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-81px)]">
        {/* Component Library Sidebar */}
        <div className="w-80 border-r border-border-subtle bg-bg-card overflow-y-auto">
          <div className="p-4">
            <h3 className="font-semibold mb-4">Component Library</h3>
            
            <Tabs defaultValue="conditions" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="conditions">Conditions</TabsTrigger>
                <TabsTrigger value="logic">Logic</TabsTrigger>
                <TabsTrigger value="actions">Actions</TabsTrigger>
              </TabsList>
              
              <TabsContent value="conditions" className="space-y-2 mt-4">
                {COMPLIANCE_COMPONENTS.conditions.map((component) => (
                  <Card
                    key={component.type}
                    className="cursor-grab active:cursor-grabbing hover:border-accent-green/50 transition-colors"
                    draggable
                    onDragStart={(e) => onDragStart(e, component)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <component.icon className="w-4 h-4 mt-0.5 text-accent-green" />
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-sm">{component.label}</h4>
                          <p className="text-xs text-text-secondary mt-1">{component.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>
              
              <TabsContent value="logic" className="space-y-2 mt-4">
                {COMPLIANCE_COMPONENTS.logic.map((component) => (
                  <Card
                    key={component.type}
                    className="cursor-grab active:cursor-grabbing hover:border-accent-green/50 transition-colors"
                    draggable
                    onDragStart={(e) => onDragStart(e, component)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <component.icon className="w-4 h-4 mt-0.5 text-accent-green" />
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-sm">{component.label}</h4>
                          <p className="text-xs text-text-secondary mt-1">{component.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>
              
              <TabsContent value="actions" className="space-y-2 mt-4">
                {COMPLIANCE_COMPONENTS.actions.map((component) => (
                  <Card
                    key={component.type}
                    className="cursor-grab active:cursor-grabbing hover:border-accent-green/50 transition-colors"
                    draggable
                    onDragStart={(e) => onDragStart(e, component)}
                  >
                    <CardContent className="p-3">
                      <div className="flex items-start gap-2">
                        <component.icon className="w-4 h-4 mt-0.5 text-accent-green" />
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-sm">{component.label}</h4>
                          <p className="text-xs text-text-secondary mt-1">{component.description}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Main Canvas */}
        <div className="flex-1 flex flex-col">
          {/* Canvas */}
          <div className="flex-1 relative group" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onInit={setReactFlowInstance}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick} // Added onPaneClick handler
              nodeTypes={nodeTypes}
              fitView
              className="bg-bg-primary"
            >
              <Background 
                variant={BackgroundVariant.Dots} 
                gap={20} 
                size={1} 
                color="var(--color-border-subtle)"
              />
              <Controls className="bg-bg-card border-border-subtle" />
              <MiniMap 
                className="bg-bg-card border-border-subtle"
                nodeColor="var(--color-accent-green)"
              />
            </ReactFlow>
            
            {/* Empty State */}
            {nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-accent-green/20 to-green-400/20 rounded-full flex items-center justify-center">
                    <Shield className="w-8 h-8 text-accent-green" />
                  </div>
                  <h3 className="text-lg font-medium mb-2">Start Building Your Compliance Rule</h3>
                  <p className="text-text-secondary max-w-md">
                    Drag components from the left sidebar to create your compliance workflow. 
                    Connect them to build complex business logic.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Canvas Stats */}
          <div className="border-t border-border-subtle bg-bg-card p-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-6">
                <span className="text-text-secondary">
                  Components: <span className="text-text-primary font-medium">{stats.totalNodes}</span>
                </span>
                <span className="text-text-secondary">
                  Conditions: <span className="text-blue-400 font-medium">{stats.conditions}</span>
                </span>
                <span className="text-text-secondary">
                  Actions: <span className="text-green-400 font-medium">{stats.actions}</span>
                </span>
                <span className="text-text-secondary">
                  Connections: <span className="text-purple-400 font-medium">{stats.connections}</span>
                </span>
              </div>
              <Badge variant="outline" className="text-xs">
                {currentRule.status?.toUpperCase() || 'DRAFT'}
              </Badge>
            </div>
          </div>
        </div>

        {/* Properties Panel */}
        <div className="w-80 border-l border-border-subtle bg-bg-card overflow-y-auto">
          <div className="p-4">
            <Tabs defaultValue="rule">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="rule">Rule Info</TabsTrigger>
                <TabsTrigger value="activity">Activity</TabsTrigger>
              </TabsList>
              
              <TabsContent value="rule" className="space-y-4 mt-4">
                <div>
                  <Label htmlFor="rule-name">Rule Name</Label>
                  <Input
                    id="rule-name"
                    value={currentRule.name || ''}
                    onChange={(e) => setCurrentRule(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="Enter rule name..."
                  />
                </div>
                
                <div>
                  <Label htmlFor="rule-description">Description</Label>
                  <Textarea
                    id="rule-description"
                    value={currentRule.description || ''}
                    onChange={(e) => setCurrentRule(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Describe this rule..."
                    rows={3}
                  />
                </div>
                
                <div>
                  <Label htmlFor="rule-category">Category</Label>
                  <Select
                    value={currentRule.category || ''}
                    onValueChange={(value) => setCurrentRule(prev => ({ ...prev, category: value as RuleCategory }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select category..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={RuleCategory.KYC}>KYC</SelectItem>
                      <SelectItem value={RuleCategory.AML}>AML</SelectItem>
                      <SelectItem value={RuleCategory.SANCTIONS}>Sanctions</SelectItem>
                      <SelectItem value={RuleCategory.PEP}>PEP</SelectItem>
                      <SelectItem value={RuleCategory.RISK_ASSESSMENT}>Risk Assessment</SelectItem>
                      <SelectItem value={RuleCategory.COMPLIANCE}>Compliance</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="rule-priority">Priority (1-10)</Label>
                  <Input
                    id="rule-priority"
                    type="number"
                    min="1"
                    max="10"
                    value={currentRule.priority || 5}
                    onChange={(e) => setCurrentRule(prev => ({ ...prev, priority: parseInt(e.target.value) || 5 }))}
                  />
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch
                    id="rule-enabled"
                    checked={currentRule.is_enabled || false}
                    onCheckedChange={(checked) => setCurrentRule(prev => ({ ...prev, is_enabled: checked }))}
                  />
                  <Label htmlFor="rule-enabled">Rule Enabled</Label>
                </div>

                {/* Selected Node Properties */}
                {selectedNode && (
                  <>
                    <Separator />
                    <div>
                      <h4 className="font-medium mb-2">Selected Component</h4>
                      <div className="p-3 bg-bg-primary rounded-lg">
                        <p className="font-medium text-sm">{selectedNode.data.label}</p>
                        <p className="text-xs text-text-secondary mt-1">{selectedNode.data.description}</p>
                        
                        {selectedNode.data.condition && (
                          <div className="mt-3 space-y-2">
                            <h5 className="text-sm font-medium">Condition</h5>
                            <div className="space-y-2 text-xs">
                              <div>
                                <Label>Field</Label>
                                <Input 
                                  value={selectedNode.data.condition.field} 
                                  className="h-7 text-xs"
                                  readOnly
                                />
                              </div>
                              <div>
                                <Label>Operator</Label>
                                <Input 
                                  value={selectedNode.data.condition.operator} 
                                  className="h-7 text-xs"
                                  readOnly
                                />
                              </div>
                              <div>
                                <Label>Value</Label>
                                <Input 
                                  value={selectedNode.data.condition.value} 
                                  className="h-7 text-xs"
                                  readOnly
                                />
                              </div>
                            </div>
                          </div>
                        )}
                        
                        {selectedNode.data.action && (
                          <div className="mt-3 space-y-2">
                            <h5 className="text-sm font-medium">Action</h5>
                            <div className="space-y-2 text-xs">
                              <div>
                                <Label>Type</Label>
                                <Input 
                                  value={selectedNode.data.action.action_type} 
                                  className="h-7 text-xs"
                                  readOnly
                                />
                              </div>
                              <div>
                                <Label>Severity</Label>
                                <Input 
                                  value={selectedNode.data.action.severity} 
                                  className="h-7 text-xs"
                                  readOnly
                                />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </TabsContent>
              
              <TabsContent value="activity" className="mt-4">
                <div>
                  <h4 className="font-medium mb-3">Activity Log</h4>
                  <div className="space-y-2">
                    {activity.length === 0 ? (
                      <p className="text-text-secondary text-sm">No activity yet</p>
                    ) : (
                      activity.map((entry, index) => (
                        <div key={index} className="p-2 bg-bg-primary rounded text-xs">
                          {entry}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}