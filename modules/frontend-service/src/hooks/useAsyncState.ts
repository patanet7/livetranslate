/**
 * Async State Hook
 *
 * Simplified state management for async operations.
 * Eliminates duplicate loading/error state patterns.
 */

import { useState, useCallback } from "react";

/**
 * State for async operations
 */
export interface AsyncState<T> {
  data: T;
  loading: boolean;
  error: string | null;
}

/**
 * Hook for managing async operation state
 *
 * Used for: API calls, file operations, async computations
 * Replaces: Duplicate loading/error state management (~50-80 lines)
 *
 * @example
 * const { data, loading, error, execute, setData } = useAsyncState<BotInstance[]>([]);
 *
 * // Execute async operation
 * useEffect(() => {
 *   execute(async () => {
 *     const response = await fetch('/api/bots');
 *     return response.json();
 *   });
 * }, [execute]);
 *
 * // Manual data update
 * setData([...data, newBot]);
 */
export const useAsyncState = <T>(initialValue: T) => {
  const [data, setData] = useState<T>(initialValue);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Execute an async operation with automatic loading/error handling
   *
   * @param asyncFn - Async function to execute
   * @returns Promise with the result
   */
  const execute = useCallback(async (asyncFn: () => Promise<T>): Promise<T> => {
    setLoading(true);
    setError(null);

    try {
      const result = await asyncFn();
      setData(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Reset to initial state
   */
  const reset = useCallback(() => {
    setData(initialValue);
    setLoading(false);
    setError(null);
  }, [initialValue]);

  /**
   * Clear error
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    data,
    loading,
    error,
    execute,
    setData,
    reset,
    clearError,
  };
};

/**
 * Hook for managing multiple async states
 *
 * @example
 * const states = useAsyncStates({
 *   bots: [] as BotInstance[],
 *   settings: defaultSettings,
 * });
 *
 * // Access individual states
 * const { loading: botsLoading } = states.bots;
 *
 * // Execute operations
 * await states.bots.execute(async () => {
 *   return fetchBots();
 * });
 */
export const useAsyncStates = <T extends Record<string, any>>(
  initialValues: T,
): { [K in keyof T]: ReturnType<typeof useAsyncState<T[K]>> } => {
  const states = {} as {
    [K in keyof T]: ReturnType<typeof useAsyncState<T[K]>>;
  };

  for (const key in initialValues) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    states[key] = useAsyncState(initialValues[key]);
  }

  return states;
};

export default useAsyncState;
