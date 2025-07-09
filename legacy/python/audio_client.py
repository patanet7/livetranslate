import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import base64
import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioClient:
    def __init__(self, server_url="ws://localhost:8765", 
                 sample_rate=16000, channels=1, blocksize=8000):
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.running = False
        self.websocket = None
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Connected to {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")
            
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio data from sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
            
        if self.websocket and self.websocket.open:
            # Convert multi-channel to mono if needed
            if self.channels == 1 and indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1).astype(np.int16)
            else:
                audio_data = indata[:, 0].astype(np.int16)
                
            # Convert to bytes and encode in base64
            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Create message
            message = json.dumps({
                "audio": audio_b64,
                "sample_rate": self.sample_rate,
                "channels": 1  # Always send mono
            })
            
            # Send asynchronously
            asyncio.create_task(self.send_message(message))
            
    async def send_message(self, message):
        """Send a message to the server"""
        if self.websocket:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection closed while sending")
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                
    async def receive_messages(self):
        """Receive and process messages from the server"""
        while self.running and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Handle transcript messages
                if "text" in data:
                    is_final = data.get("is_final", False)
                    marker = "[FINAL]" if is_final else "[partial]"
                    print(f"{marker} {data['text']}")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection closed")
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
                
    def start_microphone_streaming(self):
        """Start audio capture from microphone and streaming"""
        if self.running:
            return
            
        self.running = True
        
        # Start audio capture
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.blocksize
        )
        self.stream.start()
        logger.info(f"Started microphone capture: {self.sample_rate}Hz, {self.channels} channel(s)")
        
    def stop_streaming(self):
        """Stop audio capture and streaming"""
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Stopped audio capture")
            
    async def stream_audio_file(self, file_path, chunk_size=8000, delay_factor=1.0):
        """Stream audio from a file to the server"""
        if not await self.connect():
            return
            
        try:
            # Check if file exists
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return
                
            # Load audio file
            logger.info(f"Loading audio file: {file_path}")
            audio_data, file_sample_rate = sf.read(file_path)
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample if needed
            if file_sample_rate != self.sample_rate:
                logger.info(f"Resampling audio from {file_sample_rate}Hz to {self.sample_rate}Hz")
                # This is a very simple resampling method - for production, use a proper resampling library
                samples_per_second = len(audio_data) / (len(audio_data) / file_sample_rate)
                target_length = int(len(audio_data) * (self.sample_rate / file_sample_rate))
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length), 
                    np.arange(len(audio_data)), 
                    audio_data
                )
                
            # Convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Start the receiver task
            self.running = True
            receiver_task = asyncio.create_task(self.receive_messages())
            
            # Stream audio in chunks
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            logger.info(f"Streaming audio file in {total_chunks} chunks")
            
            for i in range(0, len(audio_data), chunk_size):
                if not self.running:
                    break
                    
                chunk = audio_data[i:i+chunk_size]
                
                # Convert to bytes and encode in base64
                audio_bytes = chunk.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Create message
                message = json.dumps({
                    "audio": audio_b64,
                    "sample_rate": self.sample_rate,
                    "channels": 1  # Always send mono
                })
                
                # Send message
                await self.send_message(message)
                
                # Calculate delay based on chunk size and sample rate
                # This simulates real-time streaming by delaying based on audio duration
                chunk_duration = len(chunk) / self.sample_rate
                delay = chunk_duration * delay_factor
                logger.debug(f"Sent chunk {i//chunk_size + 1}/{total_chunks}, sleeping for {delay:.3f}s")
                await asyncio.sleep(delay)
                
            logger.info("Finished streaming audio file")
            
            # Keep the connection open for a bit to receive final transcripts
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error streaming audio file: {e}")
        finally:
            self.running = False
            receiver_task.cancel()
            await self.disconnect()
            
    async def run_microphone_mode(self):
        """Run the client with microphone input"""
        if not await self.connect():
            return
            
        self.start_microphone_streaming()
        
        # Start message receiver
        receiver_task = asyncio.create_task(self.receive_messages())
        
        try:
            print("\n=== LiveTranslate Audio Client ===")
            print("Streaming audio from your microphone to the server")
            print("Press Ctrl+C to stop streaming\n")
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_streaming()
            await self.disconnect()
            receiver_task.cancel()

async def main():
    parser = argparse.ArgumentParser(description="LiveTranslate Audio Client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--blocksize", type=int, default=8000, help="Block size in samples")
    parser.add_argument("--file", help="Path to audio file to stream (WAV format)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (for file mode)")
    
    args = parser.parse_args()
    
    client = AudioClient(
        server_url=args.server,
        sample_rate=args.rate,
        channels=args.channels,
        blocksize=args.blocksize
    )
    
    try:
        if args.file:
            print(f"Streaming audio from file: {args.file}")
            await client.stream_audio_file(args.file, delay_factor=1.0/args.speed)
        else:
            await client.run_microphone_mode()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if not args.file:
            client.stop_streaming()
        
if __name__ == "__main__":
    asyncio.run(main()) 