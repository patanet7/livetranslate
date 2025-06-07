import asyncio
import websockets
import json
import logging
import argparse
import threading
import time
import os
import subprocess
from datetime import datetime
from typing import Set
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/logging_server.log')
    ]
)
logger = logging.getLogger(__name__)

class LoggingServer:
    def __init__(self, host="0.0.0.0", port=8766):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.log_buffer = []
        self.max_buffer_size = 1000
        self.is_running = False
        
        # Log files to monitor
        self.log_files = [
            '/tmp/transcription.log',
            '/tmp/translation.log', 
            '/tmp/whisper_npu.log',
            '/tmp/logging_server.log'
        ]
        
        # File positions for tailing
        self.file_positions = {}
        
    async def register(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Logging client connected. Total clients: {len(self.clients)}")
        
        # Send recent logs to new client
        for log_entry in self.log_buffer[-50:]:  # Send last 50 log entries
            try:
                await websocket.send(json.dumps(log_entry))
            except websockets.exceptions.ConnectionClosed:
                break
    
    async def unregister(self, websocket):
        """Unregister a client connection"""
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f"Logging client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_log(self, log_entry):
        """Send log entry to all connected clients"""
        if not self.clients:
            return
            
        # Add to buffer
        self.log_buffer.append(log_entry)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)
            
        # Broadcast to all clients
        message = json.dumps(log_entry)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending log to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister(client)
    
    def tail_log_files(self):
        """Monitor log files and send new entries"""
        for log_file in self.log_files:
            if os.path.exists(log_file):
                if log_file not in self.file_positions:
                    # Get current file size
                    self.file_positions[log_file] = os.path.getsize(log_file)
                
                try:
                    with open(log_file, 'r') as f:
                        f.seek(self.file_positions[log_file])
                        new_lines = f.readlines()
                        
                        for line in new_lines:
                            line = line.strip()
                            if line:
                                log_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "source": os.path.basename(log_file),
                                    "level": "INFO",  # Default level
                                    "message": line
                                }
                                
                                # Try to parse log level from line
                                if " - ERROR - " in line:
                                    log_entry["level"] = "ERROR"
                                elif " - WARNING - " in line:
                                    log_entry["level"] = "WARNING"
                                elif " - DEBUG - " in line:
                                    log_entry["level"] = "DEBUG"
                                
                                # Schedule broadcast
                                asyncio.create_task(self.broadcast_log(log_entry))
                        
                        # Update position
                        self.file_positions[log_file] = f.tell()
                        
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
    
    def get_docker_logs(self):
        """Get logs from Docker containers"""
        containers = [
            'livetranslate-whisper-npu',
            'livetranslate-transcription', 
            'livetranslate-translation',
            'livetranslate-frontend'
        ]
        
        for container in containers:
            try:
                # Get recent logs (last 10 lines)
                result = subprocess.run(
                    ['docker', 'logs', '--tail', '10', container],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            log_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "source": f"docker-{container}",
                                "level": "INFO",
                                "message": line.strip()
                            }
                            asyncio.create_task(self.broadcast_log(log_entry))
                
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if line.strip():
                            log_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "source": f"docker-{container}",
                                "level": "ERROR",
                                "message": line.strip()
                            }
                            asyncio.create_task(self.broadcast_log(log_entry))
                            
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout getting logs for container {container}")
            except Exception as e:
                logger.error(f"Error getting Docker logs for {container}: {e}")
    
    async def log_monitor_worker(self):
        """Background worker to monitor logs"""
        while self.is_running:
            try:
                # Monitor log files
                self.tail_log_files()
                
                # Get Docker logs every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.get_docker_logs()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in log monitor worker: {e}")
                await asyncio.sleep(5)
    
    async def handle_client(self, websocket):
        """Handle a client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle client commands
                    if data.get('command') == 'get_logs':
                        # Send recent logs
                        for log_entry in self.log_buffer[-100:]:
                            await websocket.send(json.dumps(log_entry))
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from logging client")
                except Exception as e:
                    logger.error(f"Error processing logging client message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def serve(self):
        """Start the logging server"""
        self.is_running = True
        
        # Start log monitoring worker
        monitor_task = asyncio.create_task(self.log_monitor_worker())
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Logging server started at ws://{self.host}:{self.port}")
            
            # Send initial system info
            await self.broadcast_log({
                "timestamp": datetime.now().isoformat(),
                "source": "logging_server",
                "level": "INFO",
                "message": "Logging server started - monitoring system logs"
            })
            
            try:
                await asyncio.Future()  # run forever
            except KeyboardInterrupt:
                logger.info("Logging server shutting down...")
                self.is_running = False
                monitor_task.cancel()

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="LiveTranslate Logging Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server = LoggingServer(host=args.host, port=args.port)
    
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Logging server stopped")

if __name__ == "__main__":
    main() 