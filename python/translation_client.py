import asyncio
import websockets
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationClient:
    def __init__(self, server_url="ws://localhost:8010"):
        self.server_url = server_url
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
            
    async def translate_text(self, text):
        """Send text to translate and receive the translation"""
        if not self.websocket:
            logger.error("Not connected to server")
            return None
            
        # Create message
        message = json.dumps({
            "text": text
        })
        
        try:
            # Send message
            await self.websocket.send(message)
            logger.info(f"Sent text for translation: {text[:30]}...")
            
            # Wait for response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            return data.get("translation")
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed while translating")
            return None
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            return None
    
    async def interactive_mode(self):
        """Run an interactive translation session"""
        if not await self.connect():
            return
            
        try:
            print("\n=== LiveTranslate Interactive Translation Client ===")
            print("Type text to translate (English to Chinese or Chinese to English)")
            print("Type 'exit' or press Ctrl+C to quit\n")
            
            while True:
                # Get input from user
                text = input("> ")
                
                if text.lower() in ['exit', 'quit', 'q']:
                    break
                    
                if not text.strip():
                    continue
                    
                # Translate
                translation = await self.translate_text(text)
                if translation:
                    print(f"Translation: {translation}")
                else:
                    print("Failed to get translation")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            await self.disconnect()
    
    async def stream_file(self, file_path, line_delay=1.0):
        """Stream a text file line by line to the translation server"""
        if not await self.connect():
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"\n=== Streaming file: {file_path} ===")
            print(f"Found {len(lines)} lines to translate\n")
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                print(f"Line {i+1}: {line}")
                translation = await self.translate_text(line)
                
                if translation:
                    print(f"Translation: {translation}\n")
                else:
                    print("Failed to get translation\n")
                    
                # Wait before sending next line
                await asyncio.sleep(line_delay)
                
        except Exception as e:
            logger.error(f"Error streaming file: {e}")
        finally:
            await self.disconnect()

async def main():
    parser = argparse.ArgumentParser(description="LiveTranslate Translation Client")
    parser.add_argument("--server", default="ws://localhost:8010", help="WebSocket server URL")
    parser.add_argument("--file", help="Path to a text file to translate line by line")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between lines when streaming a file (seconds)")
    
    args = parser.parse_args()
    
    client = TranslationClient(server_url=args.server)
    
    if args.file:
        await client.stream_file(args.file, args.delay)
    else:
        await client.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main()) 