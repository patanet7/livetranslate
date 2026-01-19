/**
 * Session Utility Functions
 *
 * Shared utilities for session ID generation and management
 */

/**
 * Generate a unique session ID with timestamp and random suffix
 *
 * @param prefix - Session ID prefix (e.g., 'streaming_test', 'meeting_test')
 * @returns Unique session ID string
 */
export function generateSessionId(prefix: string = "session"): string {
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 11);
  return `${prefix}_${timestamp}_${randomSuffix}`;
}

/**
 * Generate a unique request ID
 *
 * @returns Unique request ID string
 */
export function generateRequestId(): string {
  return generateSessionId("req");
}

/**
 * Generate a unique chunk ID
 *
 * @returns Unique chunk ID string
 */
export function generateChunkId(): string {
  return `chunk_${Date.now()}`;
}

/**
 * Extract timestamp from session ID
 *
 * @param sessionId - Session ID string
 * @returns Timestamp or null if invalid
 */
export function extractTimestampFromSessionId(
  sessionId: string,
): number | null {
  const parts = sessionId.split("_");
  if (parts.length >= 2) {
    const timestamp = parseInt(parts[1], 10);
    return isNaN(timestamp) ? null : timestamp;
  }
  return null;
}

/**
 * Check if session ID is valid format
 *
 * @param sessionId - Session ID to validate
 * @returns True if valid format
 */
export function isValidSessionId(sessionId: string): boolean {
  const parts = sessionId.split("_");
  return parts.length === 3 && !isNaN(parseInt(parts[1], 10));
}
