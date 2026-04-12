/**
 * MIME types for media recording.
 * VP9 is preferred for better compression, WebM with VP8 as fallback.
 */

/** VP9 codec in WebM container - best compression */
export const vp9MimeType = 'video/webm;codecs=vp9,opus';

/** VP8 codec in WebM container - wider compatibility fallback */
export const webmMimeType = 'video/webm;codecs=vp8,opus';

/** Check if a MIME type is supported by MediaRecorder */
export function isMimeTypeSupported(mimeType: string): boolean {
  if (typeof MediaRecorder === 'undefined') {
    return false;
  }
  return MediaRecorder.isTypeSupported(mimeType);
}
