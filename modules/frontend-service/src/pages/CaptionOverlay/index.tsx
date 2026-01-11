/**
 * Caption Overlay Page
 *
 * Standalone page for OBS Browser Source integration.
 * Use this URL as your browser source in OBS:
 *
 * http://localhost:5173/caption-overlay?session=YOUR_SESSION_ID
 *
 * URL Parameters:
 * - session: Session ID to connect to (required)
 * - lang: Target language filter (optional)
 * - showSpeaker: Show speaker names (default: true)
 * - showOriginal: Show original text (default: false)
 * - fontSize: Font size in pixels (default: 18)
 * - position: top | center | bottom (default: bottom)
 * - maxCaptions: Maximum visible captions (default: 3)
 * - bg: Caption background color (default: rgba(0,0,0,0.7))
 *
 * Example:
 * http://localhost:5173/caption-overlay?session=meeting-123&lang=es&fontSize=24
 */

import React, { useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { CaptionOverlay } from '../../components/CaptionOverlay';

const CaptionOverlayPage: React.FC = () => {
  const [searchParams] = useSearchParams();

  // Parse URL parameters
  const config = useMemo(() => {
    const sessionId = searchParams.get('session') || 'default';
    const targetLanguage = searchParams.get('lang') || undefined;
    const showSpeakerName = searchParams.get('showSpeaker') !== 'false';
    const showOriginal = searchParams.get('showOriginal') === 'true';
    const showConnectionStatus = searchParams.get('showStatus') === 'true';
    const fontSize = parseInt(searchParams.get('fontSize') || '18', 10);
    const position = (searchParams.get('position') || 'bottom') as 'top' | 'center' | 'bottom';
    const maxCaptions = parseInt(searchParams.get('maxCaptions') || '3', 10);
    const captionBackground = searchParams.get('bg') || 'rgba(0,0,0,0.7)';
    const animate = searchParams.get('animate') !== 'false';

    return {
      sessionId,
      targetLanguage,
      showSpeakerName,
      showOriginal,
      showConnectionStatus,
      fontSize,
      position,
      maxCaptions,
      captionBackground,
      animate,
    };
  }, [searchParams]);

  // Full-screen transparent container
  const containerStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    backgroundColor: 'transparent',
    overflow: 'hidden',
  };

  // Error message if no session
  if (!searchParams.get('session')) {
    return (
      <div style={containerStyle}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#ffffff',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            fontFamily: 'system-ui, sans-serif',
            padding: '20px',
            textAlign: 'center',
          }}
        >
          <h2 style={{ marginBottom: '20px' }}>Caption Overlay - Browser Source</h2>
          <p style={{ marginBottom: '10px' }}>Add <code>?session=YOUR_SESSION_ID</code> to the URL</p>
          <p style={{ fontSize: '14px', opacity: 0.8, marginTop: '20px' }}>
            Example: /caption-overlay?session=meeting-123&amp;lang=es&amp;fontSize=24
          </p>
          <div style={{ marginTop: '30px', textAlign: 'left', fontSize: '14px' }}>
            <p style={{ fontWeight: 'bold', marginBottom: '10px' }}>Available Parameters:</p>
            <ul style={{ listStyleType: 'none', padding: 0 }}>
              <li>• <strong>session</strong> - Session ID (required)</li>
              <li>• <strong>lang</strong> - Target language filter</li>
              <li>• <strong>showSpeaker</strong> - Show speaker names (true/false)</li>
              <li>• <strong>showOriginal</strong> - Show original text (true/false)</li>
              <li>• <strong>fontSize</strong> - Font size in pixels</li>
              <li>• <strong>position</strong> - top, center, or bottom</li>
              <li>• <strong>maxCaptions</strong> - Max visible captions</li>
              <li>• <strong>bg</strong> - Background color (CSS)</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <CaptionOverlay
        sessionId={config.sessionId}
        targetLanguage={config.targetLanguage}
        showSpeakerName={config.showSpeakerName}
        showOriginal={config.showOriginal}
        showConnectionStatus={config.showConnectionStatus}
        fontSize={config.fontSize}
        position={config.position}
        maxCaptions={config.maxCaptions}
        captionBackground={config.captionBackground}
        animate={config.animate}
      />
    </div>
  );
};

export default CaptionOverlayPage;
