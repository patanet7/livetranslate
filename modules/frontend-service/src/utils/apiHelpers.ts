/**
 * API Helper Utilities
 *
 * Centralized utilities for API calls with consistent error handling.
 * Eliminates duplicate fetch patterns across the codebase.
 */

/**
 * Standard API error class
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Fetch JSON data with automatic error handling
 *
 * @param url - API endpoint URL
 * @param options - Fetch options
 * @param errorMessage - Custom error message
 * @returns Parsed JSON response
 *
 * Used for: All API GET/POST/PUT/DELETE requests
 * Replaces: ~70 duplicate try/catch fetch blocks
 *
 * @example
 * const bot = await fetchJson<BotInstance>('/api/bot/123');
 * const result = await fetchJson('/api/settings', { method: 'POST', body: JSON.stringify(data) });
 */
export async function fetchJson<T = any>(
  url: string,
  options?: RequestInit,
  errorMessage?: string
): Promise<T> {
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      // Try to parse error response
      let errorData: any = {};
      try {
        errorData = await response.json();
      } catch {
        // Response is not JSON
      }

      const message =
        errorData.detail ||
        errorData.message ||
        errorData.error ||
        errorMessage ||
        `HTTP ${response.status}: ${response.statusText}`;

      throw new ApiError(message, response.status, errorData);
    }

    // Handle empty responses (204 No Content, etc.)
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return {} as T;
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    // Network error or other fetch failure
    const message = error instanceof Error ? error.message : String(error);
    throw new ApiError(
      errorMessage || message,
      0, // 0 indicates network error
      { originalError: error }
    );
  }
}

/**
 * Perform GET request
 *
 * @param url - API endpoint URL
 * @param errorMessage - Custom error message
 * @returns Parsed JSON response
 */
export async function get<T = any>(
  url: string,
  errorMessage?: string
): Promise<T> {
  return fetchJson<T>(url, { method: 'GET' }, errorMessage);
}

/**
 * Perform POST request
 *
 * @param url - API endpoint URL
 * @param data - Request body data
 * @param errorMessage - Custom error message
 * @returns Parsed JSON response
 */
export async function post<T = any>(
  url: string,
  data: any,
  errorMessage?: string
): Promise<T> {
  return fetchJson<T>(
    url,
    {
      method: 'POST',
      body: JSON.stringify(data),
    },
    errorMessage
  );
}

/**
 * Perform PUT request
 *
 * @param url - API endpoint URL
 * @param data - Request body data
 * @param errorMessage - Custom error message
 * @returns Parsed JSON response
 */
export async function put<T = any>(
  url: string,
  data: any,
  errorMessage?: string
): Promise<T> {
  return fetchJson<T>(
    url,
    {
      method: 'PUT',
      body: JSON.stringify(data),
    },
    errorMessage
  );
}

/**
 * Perform DELETE request
 *
 * @param url - API endpoint URL
 * @param errorMessage - Custom error message
 * @returns Parsed JSON response
 */
export async function del<T = any>(
  url: string,
  errorMessage?: string
): Promise<T> {
  return fetchJson<T>(url, { method: 'DELETE' }, errorMessage);
}

/**
 * Perform request with retry logic
 *
 * @param fn - Async function to retry
 * @param options - Retry options
 * @returns Function result
 *
 * Used for: Network-sensitive operations, health checks
 * Replaces: Duplicate retry logic in useApiClient and other files
 *
 * @example
 * const data = await withRetry(() => fetchJson('/api/bot/123'), { maxRetries: 3 });
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    backoff?: 'linear' | 'exponential';
    initialDelay?: number;
    shouldRetry?: (error: any) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    backoff = 'exponential',
    initialDelay = 1000,
    shouldRetry = (error) => {
      // Retry on network errors or 5xx server errors
      if (error instanceof ApiError) {
        return error.status === 0 || error.status >= 500;
      }
      return true;
    },
  } = options;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      // Don't retry on last attempt or if shouldRetry returns false
      if (attempt === maxRetries - 1 || !shouldRetry(error)) {
        throw error;
      }

      // Calculate delay with backoff strategy
      const delay =
        backoff === 'exponential'
          ? initialDelay * Math.pow(2, attempt)
          : initialDelay * (attempt + 1);

      console.log(`Request failed, retrying in ${delay}ms... (attempt ${attempt + 1}/${maxRetries})`);

      // Wait before next retry
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw new Error('Max retries exceeded');
}

/**
 * Upload file with progress tracking
 *
 * @param url - Upload endpoint URL
 * @param file - File to upload
 * @param onProgress - Progress callback (0-100)
 * @param additionalData - Additional form fields
 * @returns Response data
 *
 * Used for: Audio file uploads, document uploads
 */
export async function uploadFile<T = any>(
  url: string,
  file: File | Blob,
  onProgress?: (progress: number) => void,
  additionalData?: Record<string, string>
): Promise<T> {
  const formData = new FormData();
  formData.append('file', file);

  // Add additional form fields
  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    // Track upload progress
    if (onProgress) {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });
    }

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch {
          resolve(xhr.responseText as any);
        }
      } else {
        try {
          const errorData = JSON.parse(xhr.responseText);
          reject(
            new ApiError(
              errorData.detail || errorData.message || `HTTP ${xhr.status}`,
              xhr.status,
              errorData
            )
          );
        } catch {
          reject(new ApiError(`HTTP ${xhr.status}: ${xhr.statusText}`, xhr.status));
        }
      }
    });

    xhr.addEventListener('error', () => {
      reject(new ApiError('Network error', 0));
    });

    xhr.addEventListener('abort', () => {
      reject(new ApiError('Upload aborted', 0));
    });

    xhr.open('POST', url);
    xhr.send(formData);
  });
}

/**
 * Build query string from parameters
 *
 * @param params - Query parameters object
 * @returns Query string (without leading ?)
 *
 * @example
 * const query = buildQueryString({ page: 1, limit: 10 });
 * // Returns: "page=1&limit=10"
 */
export function buildQueryString(params: Record<string, any>): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach((item) => searchParams.append(key, String(item)));
      } else {
        searchParams.append(key, String(value));
      }
    }
  });

  return searchParams.toString();
}

/**
 * Check if error is a specific HTTP status code
 *
 * @param error - Error to check
 * @param status - HTTP status code
 * @returns true if error matches status
 */
export function isHttpStatus(error: any, status: number): boolean {
  return error instanceof ApiError && error.status === status;
}

/**
 * Check if error is a network error
 *
 * @param error - Error to check
 * @returns true if network error
 */
export function isNetworkError(error: any): boolean {
  return error instanceof ApiError && error.status === 0;
}

/**
 * Extract error message from various error types
 *
 * @param error - Error object
 * @param defaultMessage - Fallback message
 * @returns Error message string
 *
 * Used for: Consistent error message extraction
 */
export function getErrorMessage(error: any, defaultMessage = 'An error occurred'): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error?.message) {
    return error.message;
  }
  return defaultMessage;
}
