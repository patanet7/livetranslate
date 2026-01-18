/**
 * CaptionOverlay Component
 *
 * Transparent caption overlay for displaying real-time translated captions.
 * Designed to be used as a Browser Source in OBS Studio.
 *
 * Features:
 * - WebSocket connection to caption stream
 * - Speaker name and color support
 * - Original text display option
 * - Configurable styling (font size, position, background)
 * - Caption animations
 * - Accessibility support (ARIA labels, live regions)
 */

import React, { useEffect, useState, useRef, useCallback } from "react";

// =============================================================================
// Types
// =============================================================================

export interface Caption {
  id: string;
  original_text?: string;
  translated_text: string;
  speaker_name: string;
  speaker_color?: string;
  target_language: string;
  timestamp: string;
  duration_seconds: number;
  confidence: number;
}

export interface CaptionOverlayProps {
  /** Session ID to connect to */
  sessionId: string;
  /** WebSocket URL (defaults to relative path) */
  wsUrl?: string;
  /** Maximum number of visible captions */
  maxCaptions?: number;
  /** Show speaker name */
  showSpeakerName?: boolean;
  /** Show original text above translation */
  showOriginal?: boolean;
  /** Show connection status indicator */
  showConnectionStatus?: boolean;
  /** Font size in pixels */
  fontSize?: number;
  /** Caption position: 'top' | 'center' | 'bottom' */
  position?: "top" | "center" | "bottom";
  /** Background color for caption boxes */
  captionBackground?: string;
  /** Enable entry/exit animations */
  animate?: boolean;
  /** Target language filter */
  targetLanguage?: string;
}

// =============================================================================
// Default Props
// =============================================================================

const defaultProps: Partial<CaptionOverlayProps> = {
  maxCaptions: 3,
  showSpeakerName: true,
  showOriginal: false,
  showConnectionStatus: false,
  fontSize: 18,
  position: "bottom",
  captionBackground: "rgba(0, 0, 0, 0.7)",
  animate: true,
};

// =============================================================================
// Component
// =============================================================================

export const CaptionOverlay: React.FC<CaptionOverlayProps> = (props) => {
  const {
    sessionId,
    wsUrl,
    maxCaptions = defaultProps.maxCaptions!,
    showSpeakerName = defaultProps.showSpeakerName!,
    showOriginal = defaultProps.showOriginal!,
    showConnectionStatus = defaultProps.showConnectionStatus!,
    fontSize = defaultProps.fontSize!,
    position = defaultProps.position!,
    captionBackground = defaultProps.captionBackground!,
    animate = defaultProps.animate!,
    targetLanguage,
  } = props;

  // State
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [connectionState, setConnectionState] = useState<
    "connecting" | "connected" | "disconnected"
  >("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Build WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    if (wsUrl) {
      return `${wsUrl}/api/captions/stream/${sessionId}`;
    }
    // Build URL based on current location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    let url = `${protocol}//${host}/api/captions/stream/${sessionId}`;
    if (targetLanguage) {
      url += `?target_language=${targetLanguage}`;
    }
    return url;
  }, [sessionId, wsUrl, targetLanguage]);

  // Handle incoming messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.event) {
          case "connected":
            // Load initial captions
            if (data.current_captions) {
              setCaptions(data.current_captions.slice(-maxCaptions));
            }
            break;

          case "caption_added":
            setCaptions((prev) => {
              const newCaptions = [...prev, data.caption];
              return newCaptions.slice(-maxCaptions);
            });
            break;

          case "caption_expired":
            setCaptions((prev) => prev.filter((c) => c.id !== data.caption_id));
            break;

          case "caption_updated":
            setCaptions((prev) =>
              prev.map((c) => (c.id === data.caption.id ? data.caption : c)),
            );
            break;

          case "session_cleared":
            setCaptions([]);
            break;

          case "ping":
            wsRef.current?.send(JSON.stringify({ event: "pong" }));
            break;
        }
      } catch (error) {
        console.error("Error parsing caption message:", error);
      }
    },
    [maxCaptions],
  );

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState("connecting");
    const url = getWebSocketUrl();
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnectionState("connected");
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setConnectionState("disconnected");
      // Attempt reconnection after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    wsRef.current = ws;
  }, [getWebSocketUrl, handleMessage]);

  // Setup WebSocket connection
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  // Position styles
  const getPositionStyle = (): React.CSSProperties => {
    switch (position) {
      case "top":
        return { alignItems: "flex-start", paddingTop: "20px" };
      case "center":
        return { alignItems: "center" };
      case "bottom":
      default:
        return { alignItems: "flex-end", paddingBottom: "20px" };
    }
  };

  // Container styles
  const containerStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "transparent",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    padding: "20px",
    boxSizing: "border-box",
    pointerEvents: "none",
    ...getPositionStyle(),
  };

  // Caption box styles
  const captionBoxStyle: React.CSSProperties = {
    backgroundColor: captionBackground,
    borderRadius: "8px",
    padding: "12px 16px",
    marginBottom: "8px",
    maxWidth: "80%",
    transition: animate ? "all 0.3s ease" : "none",
    animation: animate ? "captionFadeIn 0.3s ease" : "none",
  };

  // Caption text styles
  const captionTextStyle: React.CSSProperties = {
    color: "#ffffff",
    fontSize: `${fontSize}px`,
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontWeight: 500,
    lineHeight: 1.4,
    textShadow: "2px 2px 4px rgba(0, 0, 0, 0.8)",
    margin: 0,
  };

  // Speaker name styles
  const speakerStyle = (color: string = "#ffffff"): React.CSSProperties => ({
    color,
    fontSize: `${fontSize - 2}px`,
    fontWeight: 600,
    marginBottom: "4px",
    textShadow: "1px 1px 3px rgba(0, 0, 0, 0.8)",
  });

  // Original text styles
  const originalTextStyle: React.CSSProperties = {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: `${fontSize - 2}px`,
    fontStyle: "italic",
    marginBottom: "4px",
    textShadow: "1px 1px 3px rgba(0, 0, 0, 0.8)",
  };

  // Connection status styles
  const statusStyle: React.CSSProperties = {
    position: "absolute",
    top: "10px",
    right: "10px",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "12px",
    backgroundColor:
      connectionState === "connected"
        ? "rgba(76, 175, 80, 0.8)"
        : connectionState === "connecting"
          ? "rgba(255, 193, 7, 0.8)"
          : "rgba(244, 67, 54, 0.8)",
    color: "#ffffff",
  };

  return (
    <>
      {/* CSS Keyframes */}
      <style>
        {`
          @keyframes captionFadeIn {
            from {
              opacity: 0;
              transform: translateY(10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>

      <div style={containerStyle} role="region" aria-label="Live captions">
        {/* Connection Status */}
        {showConnectionStatus && (
          <div style={statusStyle}>
            {connectionState === "connected"
              ? "Connected"
              : connectionState === "connecting"
                ? "Connecting..."
                : "Disconnected"}
          </div>
        )}

        {/* Captions */}
        <div
          aria-live="polite"
          aria-atomic="false"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            width: "100%",
          }}
        >
          {captions.map((caption) => (
            <div key={caption.id} style={captionBoxStyle}>
              {/* Speaker Name */}
              {showSpeakerName && caption.speaker_name && (
                <div style={speakerStyle(caption.speaker_color)}>
                  {caption.speaker_name}
                </div>
              )}

              {/* Original Text */}
              {showOriginal && caption.original_text && (
                <div style={originalTextStyle}>{caption.original_text}</div>
              )}

              {/* Translated Text */}
              <p style={captionTextStyle}>{caption.translated_text}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  );
};

export default CaptionOverlay;
