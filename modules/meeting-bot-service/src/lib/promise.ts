import { WaitPromise } from '../types';

/**
 * Creates a promise that can be resolved externally before the timeout.
 * Useful for waiting on operations that may complete early (e.g., meeting ending).
 *
 * @param timeoutMs - Maximum time to wait in milliseconds
 * @returns A WaitPromise with the promise and early resolution function
 */
export function getWaitingPromise(timeoutMs: number): WaitPromise {
  let resolveEarly: (value: void | PromiseLike<void>) => void;

  const promise = new Promise<void>((resolve) => {
    resolveEarly = resolve;

    // Auto-resolve after timeout
    setTimeout(() => {
      resolve();
    }, timeoutMs);
  });

  return {
    promise,
    resolveEarly: resolveEarly!,
  };
}
