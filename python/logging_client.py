import asyncio
import websockets
import json
import csv
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTranslateLogger:
    def __init__(self, 
                 transcription_url="ws://localhost:8765",
                 translation_url="ws://localhost:8010",
                 output_dir="logs"):
        self.transcription_url = transcription_url
        self.translation_url = translation_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcription_file = self.output_dir / f"transcriptions_{timestamp}.csv"
        self.translation_file = self.output_dir / f"translations_{timestamp}.csv"
        
        # WebSocket connections
        self.transcription_ws = None
        self.translation_ws = None
        
        # Running state
        self.running = False
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Transcription CSV headers
        with open(self.transcription_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'text', 'is_final', 'confidence'])
            
        # Translation CSV headers  
        with open(self.translation_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'original_text', 'translated_text', 'source_lang', 'target_lang'])
            
        logger.info(f"Initialized CSV files:")
        logger.info(f"  Transcriptions: {self.transcription_file}")
        logger.info(f"  Translations: {self.translation_file}")
        
    async def log_transcription(self, data):
        """Log transcription data to CSV"""
        timestamp = datetime.now().isoformat()
        text = data.get('text', '')
        is_final = data.get('is_final', False)
        confidence = data.get('confidence', '')
        
        # Write to CSV file
        with open(self.transcription_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, text, is_final, confidence])
            
        # Print to console
        marker = "[FINAL]" if is_final else "[partial]"
        print(f"{marker} TRANSCRIPTION: {text}")
        
    async def log_translation(self, data):
        """Log translation data to CSV"""
        timestamp = datetime.now().isoformat()
        original = data.get('original', '')
        translation = data.get('translation', '')
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'auto')
        
        # Write to CSV file
        with open(self.translation_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, original, translation, source_lang, target_lang])
            
        # Print to console
        print(f"[TRANSLATION] {original} -> {translation}")
        
    async def connect_transcription_server(self):
        """Connect to transcription WebSocket server"""
        try:
            self.transcription_ws = await websockets.connect(self.transcription_url)
            logger.info(f"Connected to transcription server: {self.transcription_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to transcription server: {e}")
            return False
            
    async def connect_translation_server(self):
        """Connect to translation WebSocket server"""
        try:
            self.translation_ws = await websockets.connect(self.translation_url)
            logger.info(f"Connected to translation server: {self.translation_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to translation server: {e}")
            return False
            
    async def handle_transcription_messages(self):
        """Handle incoming transcription messages"""
        while self.running and self.transcription_ws:
            try:
                message = await self.transcription_ws.recv()
                data = json.loads(message)
                
                # Log transcription data
                await self.log_transcription(data)
                
                # If final transcription, send to translation server
                if data.get('is_final', False) and data.get('text', '').strip():
                    if self.translation_ws:
                        translation_request = {
                            "text": data['text']
                        }
                        await self.translation_ws.send(json.dumps(translation_request))
                        
            except websockets.exceptions.ConnectionClosed:
                logger.error("Transcription WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error handling transcription message: {e}")
                
    async def handle_translation_messages(self):
        """Handle incoming translation messages"""
        while self.running and self.translation_ws:
            try:
                message = await self.translation_ws.recv()
                data = json.loads(message)
                
                # Log translation data
                await self.log_translation(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.error("Translation WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error handling translation message: {e}")
                
    async def run(self):
        """Main run loop"""
        logger.info("Starting LiveTranslate Logger...")
        
        # Connect to both servers
        transcription_connected = await self.connect_transcription_server()
        translation_connected = await self.connect_translation_server()
        
        if not transcription_connected:
            logger.error("Could not connect to transcription server. Exiting.")
            return
            
        if not translation_connected:
            logger.warning("Could not connect to translation server. Only logging transcriptions.")
            
        self.running = True
        
        # Start message handling tasks
        tasks = []
        
        if transcription_connected:
            tasks.append(asyncio.create_task(self.handle_transcription_messages()))
            
        if translation_connected:
            tasks.append(asyncio.create_task(self.handle_translation_messages()))
            
        try:
            print("\n=== LiveTranslate Logger Started ===")
            print("Logging transcriptions and translations to CSV files")
            print("Press Ctrl+C to stop logging\n")
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            print("\nStopping logger...")
        finally:
            self.running = False
            
            # Close WebSocket connections
            if self.transcription_ws:
                await self.transcription_ws.close()
            if self.translation_ws:
                await self.translation_ws.close()
                
            # Cancel remaining tasks
            for task in tasks:
                task.cancel()
                
            logger.info("Logger stopped")
            
    def print_summary(self):
        """Print summary of logged data"""
        try:
            # Count transcriptions
            transcription_count = 0
            if self.transcription_file.exists():
                with open(self.transcription_file, 'r', encoding='utf-8') as f:
                    transcription_count = sum(1 for line in f) - 1  # Subtract header
                    
            # Count translations
            translation_count = 0
            if self.translation_file.exists():
                with open(self.translation_file, 'r', encoding='utf-8') as f:
                    translation_count = sum(1 for line in f) - 1  # Subtract header
                    
            print(f"\n=== Session Summary ===")
            print(f"Transcriptions logged: {transcription_count}")
            print(f"Translations logged: {translation_count}")
            print(f"Files saved:")
            print(f"  {self.transcription_file}")
            print(f"  {self.translation_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

async def main():
    parser = argparse.ArgumentParser(description="LiveTranslate Logger - Log transcriptions and translations to CSV")
    parser.add_argument("--transcription-server", default="ws://localhost:8765", 
                       help="Transcription WebSocket server URL")
    parser.add_argument("--translation-server", default="ws://localhost:8010",
                       help="Translation WebSocket server URL") 
    parser.add_argument("--output-dir", default="logs",
                       help="Directory to save CSV log files")
    
    args = parser.parse_args()
    
    logger_client = LiveTranslateLogger(
        transcription_url=args.transcription_server,
        translation_url=args.translation_server,
        output_dir=args.output_dir
    )
    
    try:
        await logger_client.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger_client.print_summary()

if __name__ == "__main__":
    asyncio.run(main()) 