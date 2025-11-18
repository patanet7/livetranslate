/**
 * HTTP Constants
 *
 * Centralized HTTP methods, status codes, and headers.
 * Eliminates magic strings and numbers across the codebase.
 */

/**
 * HTTP Methods
 *
 * Used for: fetch() method parameter, API client methods
 * Replaces: String literals 'GET', 'POST', etc.
 */
export const HTTP_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  PATCH: 'PATCH',
  DELETE: 'DELETE',
  HEAD: 'HEAD',
  OPTIONS: 'OPTIONS',
} as const;

export type HttpMethod = typeof HTTP_METHODS[keyof typeof HTTP_METHODS];

/**
 * HTTP Status Codes
 *
 * Used for: Response validation, error handling
 * Replaces: Magic numbers 200, 404, 500, etc.
 *
 * @example
 * if (response.status === HTTP_STATUS.NOT_FOUND) {
 *   console.error('Resource not found');
 * }
 */
export const HTTP_STATUS = {
  // 2xx Success
  OK: 200,
  CREATED: 201,
  ACCEPTED: 202,
  NO_CONTENT: 204,

  // 3xx Redirection
  MOVED_PERMANENTLY: 301,
  FOUND: 302,
  NOT_MODIFIED: 304,

  // 4xx Client Errors
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  METHOD_NOT_ALLOWED: 405,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,

  // 5xx Server Errors
  INTERNAL_SERVER_ERROR: 500,
  NOT_IMPLEMENTED: 501,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504,
} as const;

export type HttpStatus = typeof HTTP_STATUS[keyof typeof HTTP_STATUS];

/**
 * Content Types
 *
 * Used for: Request/Response headers
 * Replaces: String literals for Content-Type
 */
export const CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  FORM_URLENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  HTML: 'text/html',
  XML: 'application/xml',
  BINARY: 'application/octet-stream',
  PDF: 'application/pdf',
  AUDIO_WAV: 'audio/wav',
  AUDIO_MP3: 'audio/mpeg',
  AUDIO_WEBM: 'audio/webm',
} as const;

export type ContentType = typeof CONTENT_TYPES[keyof typeof CONTENT_TYPES];

/**
 * Common HTTP Headers
 *
 * Used for: Request/Response header names
 */
export const HTTP_HEADERS = {
  CONTENT_TYPE: 'Content-Type',
  AUTHORIZATION: 'Authorization',
  ACCEPT: 'Accept',
  CACHE_CONTROL: 'Cache-Control',
  USER_AGENT: 'User-Agent',
  REFERER: 'Referer',
  ACCEPT_LANGUAGE: 'Accept-Language',
  ACCEPT_ENCODING: 'Accept-Encoding',
} as const;

export type HttpHeader = typeof HTTP_HEADERS[keyof typeof HTTP_HEADERS];

/**
 * Check if status code is successful (2xx)
 */
export const isSuccessStatus = (status: number): boolean => {
  return status >= 200 && status < 300;
};

/**
 * Check if status code is client error (4xx)
 */
export const isClientError = (status: number): boolean => {
  return status >= 400 && status < 500;
};

/**
 * Check if status code is server error (5xx)
 */
export const isServerError = (status: number): boolean => {
  return status >= 500 && status < 600;
};

/**
 * Get status code description
 */
export const getStatusDescription = (status: number): string => {
  const statusMap: Record<number, string> = {
    200: 'OK',
    201: 'Created',
    204: 'No Content',
    400: 'Bad Request',
    401: 'Unauthorized',
    403: 'Forbidden',
    404: 'Not Found',
    409: 'Conflict',
    422: 'Unprocessable Entity',
    500: 'Internal Server Error',
    502: 'Bad Gateway',
    503: 'Service Unavailable',
  };

  return statusMap[status] || 'Unknown Status';
};
