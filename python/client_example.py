import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import base64
import argparse

class AudioStreamClient:
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
            print(f"Connected to {self.server_url}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")
            
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio data from sounddevice"""
        if status:
            print(f"Audio callback status: {status}")
            
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
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed while sending")
            except Exception as e:
                print(f"Error sending message: {e}")
                
    async def receive_messages(self):
        """Receive and process messages from the server"""
        while self.running and self.websocket and self.websocket.open:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Handle transcript messages
                if "text" in data:
                    is_final = data.get("is_final", False)
                    marker = "[FINAL]" if is_final else "[partial]"
                    print(f"{marker} {data['text']}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
                
    def start_streaming(self):
        """Start audio capture and streaming"""
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
        print(f"Started audio capture: {self.sample_rate}Hz, {self.channels} channel(s)")
        
    def stop_streaming(self):
        """Stop audio capture and streaming"""
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            print("Stopped audio capture")
            
    async def run(self):
        """Run the client"""
        if not await self.connect():
            return
            
        self.start_streaming()
        
        # Start message receiver
        receiver_task = asyncio.create_task(self.receive_messages())
        
        try:
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop_streaming()
            await self.disconnect()
            receiver_task.cancel()
            
async def main():
    parser = argparse.ArgumentParser(description="LiveTranslate Audio Streaming Client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--blocksize", type=int, default=8000, help="Block size in samples")
    
    args = parser.parse_args()
    
    client = AudioStreamClient(
        server_url=args.server,
        sample_rate=args.rate,
        channels=args.channels,
        blocksize=args.blocksize
    )
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        client.stop_streaming()
        
if __name__ == "__main__":
    asyncio.run(main()) 