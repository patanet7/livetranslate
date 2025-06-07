import asyncio
import websockets
import json
import numpy as np
import base64
import sys
import os
from typing import Set
import logging
import argparse

# Add the current directory to the Python path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ASR processor (required)
from asr import ASRProcessor

# Try to import AudioCapture, but make it optional for WebSocket-only mode
try:
    from capture import AudioCapture
    AUDIO_CAPTURE_AVAILABLE = True
except ImportError as e:
    AUDIO_CAPTURE_AVAILABLE = False
    logger.warning(f"AudioCapture not available: {e}. Running in WebSocket-only mode.")

class TranscriptionServer:
    def __init__(self, host: str = "localhost", port: int = 8765, passthrough: bool = False):
        self.host = host
        self.port = port
        self.passthrough = passthrough
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.asr = None
        self.capture = None
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_transcript(self, text: str, is_final: bool):
        """Send transcript to all connected clients"""
        if not self.clients:
            return
            
        message = json.dumps({
            "text": text,
            "is_final": is_final
        })
        
        # Send to all connected clients
        websockets.broadcast(self.clients, message)
        
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)
                    
                    # Handle audio data from client
                    if "audio" in data:
                        # Decode base64 audio data
                        audio_bytes = base64.b64decode(data["audio"])
                        
                        # Convert to numpy array (assuming 16-bit PCM)
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Process the audio data
                        if self.asr:
                            self.asr.process_audio(audio_data)
                            
                    # Handle configuration messages
                    elif "config" in data:
                        logger.info(f"Received config: {data['config']}")
                        # Handle config updates if needed
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    def start_transcription(self):
        """Start audio capture and transcription"""
        # Initialize ASR with callback to send transcripts
        self.asr = ASRProcessor(
            on_transcript=lambda text, is_final: 
                asyncio.get_event_loop().create_task(
                    self.send_transcript(text, is_final)
                )
        )
        
        # When using WebSockets for audio input, we don't need to capture from local devices
        # but we'll keep the option for local capture if AudioCapture is available
        self.local_capture = False
        
        if self.local_capture and AUDIO_CAPTURE_AVAILABLE:
            # Initialize audio capture with ASR processing
            self.capture = AudioCapture(
                callback=self.asr.process_audio,
                passthrough=self.passthrough
            )
            self.capture.start_capture()
            logger.info("Started local audio capture and transcription")
            if self.passthrough:
                logger.info("Audio passthrough enabled - captured audio will be played back")
        elif self.local_capture and not AUDIO_CAPTURE_AVAILABLE:
            logger.warning("Local audio capture requested but AudioCapture not available")
        else:
            logger.info("Running in WebSocket-only mode (no local audio capture)")
        
    def stop_transcription(self):
        """Stop audio capture and transcription"""
        if self.capture:
            self.capture.stop_capture()
        if self.asr:
            self.asr.stop()
        logger.info("Stopped audio capture and transcription")
        
    async def serve(self):
        """Start the WebSocket server"""
        self.start_transcription()
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveTranslate Real-time Transcription Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the server to")
    parser.add_argument("--passthrough", action="store_true", help="Enable audio passthrough to output device")
    parser.add_argument("--local-capture", action="store_true", help="Enable local audio capture")
    
    args = parser.parse_args()
    
    server = TranscriptionServer(host=args.host, port=args.port, passthrough=args.passthrough)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        server.stop_transcription()
        logger.info("Server stopped") 