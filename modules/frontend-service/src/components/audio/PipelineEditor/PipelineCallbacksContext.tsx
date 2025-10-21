import { createContext, useContext } from 'react';

/**
 * Context for providing pipeline node callbacks
 * This avoids storing callbacks in node.data which React Flow strips during state updates
 */
export interface PipelineCallbacks {
  onNodeSettingsOpen: (nodeId: string) => void;
  onGainChange: (nodeId: string, type: 'in' | 'out', value: number) => void;
  onParameterChange: (nodeId: string, paramName: string, value: number) => void;
  onToggleEnabled: (nodeId: string, enabled: boolean) => void;
  websocket: WebSocket | null;
  isRealtimeActive: boolean;
}

const PipelineCallbacksContext = createContext<PipelineCallbacks | undefined>(undefined);

export const PipelineCallbacksProvider = PipelineCallbacksContext.Provider;

export const usePipelineCallbacks = () => {
  const context = useContext(PipelineCallbacksContext);
  if (!context) {
    throw new Error('usePipelineCallbacks must be used within PipelineCallbacksProvider');
  }
  return context;
};
