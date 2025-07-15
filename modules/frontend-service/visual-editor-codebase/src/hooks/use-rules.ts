import { useState, useCallback, useRef } from 'react';

export interface Rule {
  id: string;
  name: string;
  description: string;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
  conditions: Array<{
    id: string;
    field: string;
    operator: string;
    value: any;
  }>;
  actions: Array<{
    id: string;
    type: string;
    parameters: Record<string, any>;
  }>;
  metadata?: Record<string, any>;
}

interface UseRulesReturn {
  rules: Rule[];
  loading: boolean;
  error: string | null;
  createRule: (rule: Partial<Rule>) => Promise<Rule>;
  updateRule: (id: string, updates: Partial<Rule>) => Promise<Rule>;
  deleteRule: (id: string) => Promise<void>;
  refreshRules: () => Promise<void>;
}

const MOCK_DELAY = 800;

const generateId = (): string => {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

const mockApiCall = <T>(data: T, delay: number = MOCK_DELAY): Promise<T> => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (Math.random() < 0.05) {
        reject(new Error('Network error - please try again'));
      } else {
        resolve(data);
      }
    }, delay);
  });
};

const createMockRule = (partial: Partial<Rule>): Rule => {
  const now = new Date();
  return {
    id: generateId(),
    name: 'Untitled Rule',
    description: '',
    isActive: true,
    createdAt: now,
    updatedAt: now,
    conditions: [],
    actions: [],
    metadata: {},
    ...partial,
  };
};

const initialMockRules: Rule[] = [
  {
    id: '1',
    name: 'Welcome Email Rule',
    description: 'Send welcome email to new users',
    isActive: true,
    createdAt: new Date('2024-01-15T10:00:00Z'),
    updatedAt: new Date('2024-01-15T10:00:00Z'),
    conditions: [
      {
        id: 'c1',
        field: 'user.registrationDate',
        operator: 'equals',
        value: 'today'
      }
    ],
    actions: [
      {
        id: 'a1',
        type: 'send_email',
        parameters: {
          template: 'welcome',
          recipient: '{{user.email}}'
        }
      }
    ],
    metadata: {
      category: 'user_onboarding',
      priority: 'high'
    }
  },
  {
    id: '2',
    name: 'Discount Validation',
    description: 'Apply discount for orders over $100',
    isActive: true,
    createdAt: new Date('2024-01-14T15:30:00Z'),
    updatedAt: new Date('2024-01-16T09:45:00Z'),
    conditions: [
      {
        id: 'c2',
        field: 'order.total',
        operator: 'greater_than',
        value: 100
      }
    ],
    actions: [
      {
        id: 'a2',
        type: 'apply_discount',
        parameters: {
          type: 'percentage',
          value: 10
        }
      }
    ],
    metadata: {
      category: 'promotions',
      priority: 'medium'
    }
  },
  {
    id: '3',
    name: 'Security Alert',
    description: 'Alert on suspicious login attempts',
    isActive: false,
    createdAt: new Date('2024-01-10T08:20:00Z'),
    updatedAt: new Date('2024-01-12T14:15:00Z'),
    conditions: [
      {
        id: 'c3',
        field: 'login.failedAttempts',
        operator: 'greater_than',
        value: 5
      }
    ],
    actions: [
      {
        id: 'a3',
        type: 'send_alert',
        parameters: {
          level: 'warning',
          message: 'Suspicious login activity detected'
        }
      }
    ],
    metadata: {
      category: 'security',
      priority: 'critical'
    }
  }
];

export function useRules(): UseRulesReturn {
  const [rules, setRules] = useState<Rule[]>(initialMockRules);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const loadingOperations = useRef<Set<string>>(new Set());

  const setOperationLoading = useCallback((operation: string, isLoading: boolean) => {
    if (isLoading) {
      loadingOperations.current.add(operation);
    } else {
      loadingOperations.current.delete(operation);
    }
    setLoading(loadingOperations.current.size > 0);
  }, []);

  const handleError = useCallback((error: unknown, operation: string) => {
    console.error(`Error during ${operation}:`, error);
    const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
    setError(errorMessage);
    
    setTimeout(() => {
      setError(null);
    }, 5000);
  }, []);

  const createRule = useCallback(async (ruleData: Partial<Rule>): Promise<Rule> => {
    const operationId = `create-${Date.now()}`;
    setOperationLoading(operationId, true);
    setError(null);

    try {
      const newRule = createMockRule(ruleData);
      
      setRules(prevRules => [newRule, ...prevRules]);
      
      const createdRule = await mockApiCall(newRule);
      
      setRules(prevRules => 
        prevRules.map(rule => 
          rule.id === newRule.id ? createdRule : rule
        )
      );

      return createdRule;
    } catch (error) {
      setRules(prevRules => prevRules.filter(rule => rule.id !== ruleData.id));
      handleError(error, 'create rule');
      throw error;
    } finally {
      setOperationLoading(operationId, false);
    }
  }, [setOperationLoading, handleError]);

  const updateRule = useCallback(async (id: string, updates: Partial<Rule>): Promise<Rule> => {
    const operationId = `update-${id}`;
    setOperationLoading(operationId, true);
    setError(null);

    const originalRule = rules.find(rule => rule.id === id);
    if (!originalRule) {
      const error = new Error('Rule not found');
      handleError(error, 'update rule');
      setOperationLoading(operationId, false);
      throw error;
    }

    try {
      const optimisticUpdate = {
        ...originalRule,
        ...updates,
        updatedAt: new Date()
      };

      setRules(prevRules =>
        prevRules.map(rule =>
          rule.id === id ? optimisticUpdate : rule
        )
      );

      const updatedRule = await mockApiCall(optimisticUpdate);
      
      setRules(prevRules =>
        prevRules.map(rule =>
          rule.id === id ? updatedRule : rule
        )
      );

      return updatedRule;
    } catch (error) {
      setRules(prevRules =>
        prevRules.map(rule =>
          rule.id === id ? originalRule : rule
        )
      );
      handleError(error, 'update rule');
      throw error;
    } finally {
      setOperationLoading(operationId, false);
    }
  }, [rules, setOperationLoading, handleError]);

  const deleteRule = useCallback(async (id: string): Promise<void> => {
    const operationId = `delete-${id}`;
    setOperationLoading(operationId, true);
    setError(null);

    const ruleToDelete = rules.find(rule => rule.id === id);
    if (!ruleToDelete) {
      const error = new Error('Rule not found');
      handleError(error, 'delete rule');
      setOperationLoading(operationId, false);
      throw error;
    }

    try {
      setRules(prevRules => prevRules.filter(rule => rule.id !== id));
      
      await mockApiCall(undefined);
    } catch (error) {
      setRules(prevRules => [...prevRules, ruleToDelete]);
      handleError(error, 'delete rule');
      throw error;
    } finally {
      setOperationLoading(operationId, false);
    }
  }, [rules, setOperationLoading, handleError]);

  const refreshRules = useCallback(async (): Promise<void> => {
    const operationId = 'refresh';
    setOperationLoading(operationId, true);
    setError(null);

    try {
      const refreshedRules = await mockApiCall([...initialMockRules]);
      setRules(refreshedRules);
    } catch (error) {
      handleError(error, 'refresh rules');
      throw error;
    } finally {
      setOperationLoading(operationId, false);
    }
  }, [setOperationLoading, handleError]);

  return {
    rules,
    loading,
    error,
    createRule,
    updateRule,
    deleteRule,
    refreshRules,
  };
}