#!/usr/bin/env python3
"""
Simple vLLM Translation Server

Direct implementation using vLLM for real translation without complex dependencies.
Based on the working legacy server but integrated into translation-service.
Enhanced with REST endpoints for service integration.
"""

import asyncio
import websockets
import json
import logging
import time
import threading
from datetime import datetime
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from typing import Dict, Optional
import uuid

# Direct vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationHTTPHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP handler for health checks and REST API"""
    
    def __init__(self, server_instance, *args, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/languages':
            self._handle_languages()
        elif self.path == '/stats':
            self._handle_stats()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/translate':
            self._handle_translate()
        elif self.path == '/translate/batch':
            self._handle_batch_translate()
        else:
            self.send_response(404)
            self.end_headers()
    
    def _handle_health(self):
        """Health check endpoint"""
        if self.server_instance.is_model_ready:
            self.send_response(200)
            response = {
                "status": "healthy",
                "model_ready": True,
                "clients": len(self.server_instance.clients),
                "model": self.server_instance.model_name,
                "version": "1.0.0",
                "endpoints": {
                    "websocket": f"ws://{self.server_instance.host}:{self.server_instance.port}",
                    "translate": f"http://{self.server_instance.host}:{self.server_instance.port}/translate",
                    "languages": f"http://{self.server_instance.host}:{self.server_instance.port}/languages"
                }
            }
        else:
            self.send_response(503)
            response = {
                "status": "loading",
                "model_ready": False,
                "message": "Model is still loading"
            }
        
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_languages(self):
        """Supported languages endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "supported_languages": {
                "zh": "Chinese",
                "en": "English"
            },
            "auto_detect": True,
            "bidirectional": True
        }
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_stats(self):
        """Statistics endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        stats = getattr(self.server_instance, 'stats', {
            "translations_processed": 0,
            "clients_connected": len(self.server_instance.clients),
            "model_ready": self.server_instance.is_model_ready
        })
        self.wfile.write(json.dumps(stats).encode())
    
    def _handle_translate(self):
        """Single translation endpoint"""
        if not self.server_instance.is_model_ready:
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Model not ready"}).encode())
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            if not text:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'text' field"}).encode())
                return
            
            # Translate
            result = self.server_instance.translate_text(text)
            
            # Add request metadata
            result['request_id'] = str(uuid.uuid4())
            result['timestamp'] = datetime.utcnow().isoformat()
            
            # Update stats
            if hasattr(self.server_instance, 'stats'):
                self.server_instance.stats['translations_processed'] = \
                    self.server_instance.stats.get('translations_processed', 0) + 1
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def _handle_batch_translate(self):
        """Batch translation endpoint"""
        if not self.server_instance.is_model_ready:
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Model not ready"}).encode())
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            texts = data.get('texts', [])
            if not texts:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'texts' field"}).encode())
                return
            
            # Translate all texts
            results = []
            for i, text in enumerate(texts):
                result = self.server_instance.translate_text(text)
                result['index'] = i
                results.append(result)
            
            response = {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "results": results,
                "total_texts": len(texts),
                "successful": len([r for r in results if 'error' not in r])
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default HTTP server logs
        pass

class SimpleVLLMTranslationServer:
    """Simple vLLM translation server with real AI translation"""
    
    def __init__(self, host="0.0.0.0", port=8010, model_name="Qwen/Qwen2.5-14B-Instruct-AWQ"):
        self.host = host
        self.port = port
        self.clients = set()
        self.model_name = model_name
        self.lock = threading.Lock()
        self.is_model_ready = False
        
        # Statistics tracking
        self.stats = {
            "translations_processed": 0,
            "clients_connected": 0,
            "model_ready": False,
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Initialize model in background thread
        self.init_thread = threading.Thread(target=self._initialize_model)
        self.init_thread.daemon = True
        self.init_thread.start()
        
        # Start health check server
        self.health_thread = threading.Thread(target=self._start_health_server)
        self.health_thread.daemon = True
        self.health_thread.start()
        
        logger.info(f"Starting vLLM Translation Server")
        logger.info(f"Model: {model_name}")
        logger.info(f"WebSocket: ws://{host}:{port}")
        logger.info(f"REST API: http://{host}:{port}")
        logger.info(f"Health Check: http://{host}:{port}/health")
    
    def _start_health_server(self):
        """Start HTTP server for health checks and REST API"""
        try:
            def handler(*args, **kwargs):
                return TranslationHTTPHandler(self, *args, **kwargs)
            
            httpd = HTTPServer((self.host, self.port), handler)
            logger.info(f"REST API server running on http://{self.host}:{self.port}")
            logger.info(f"Available endpoints:")
            logger.info(f"  GET  /health - Service health check")
            logger.info(f"  GET  /languages - Supported languages")
            logger.info(f"  GET  /stats - Usage statistics")
            logger.info(f"  POST /translate - Single translation")
            logger.info(f"  POST /translate/batch - Batch translation")
            httpd.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
    
    def _initialize_model(self):
        """Initialize vLLM model and tokenizer"""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Optimized sampling parameters for translation
            self.sampling_params = SamplingParams(
                temperature=0.3,  # Lower for more consistent translations
                top_p=0.9,
                top_k=50,
                max_tokens=512,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )
            
            logger.info(f"Loading vLLM model: {self.model_name}")
            logger.info("This may take a few minutes for first-time download...")
            
            self.llm = LLM(
                model=self.model_name,
                quantization="awq",  # Use AWQ quantization
                dtype="half",
                tensor_parallel_size=1,
                max_model_len=2048,
                gpu_memory_utilization=0.9,
                trust_remote_code=True
            )
            
            with self.lock:
                self.is_model_ready = True
                self.stats['model_ready'] = True
            
            logger.info("✅ vLLM translation model ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            traceback.print_exc()
    
    def detect_language(self, text):
        """Simple language detection"""
        # Count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len([c for c in text if c.isalnum()])
        
        if total_chars == 0:
            return 'en'
        
        # If more than 30% Chinese characters, consider it Chinese
        if chinese_chars / total_chars > 0.3:
            return 'zh'
        else:
            return 'en'
    
    def create_prompt(self, text, source_lang, target_lang):
        """Create translation prompt"""
        if source_lang == 'zh' and target_lang == 'en':
            system = "You are an expert translator. Translate the Chinese text to natural, fluent English. Return only the English translation."
        elif source_lang == 'en' and target_lang == 'zh':
            system = "You are an expert translator. Translate the English text to natural, fluent Chinese. Return only the Chinese translation."
        else:
            system = "You are an expert translator. Translate the text accurately. Return only the translation."
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate: {text}"}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def translate_text(self, input_text):
        """Translate text using vLLM"""
        start_time = time.time()
        
        with self.lock:
            if not self.is_model_ready:
                return {
                    "translation": "Translation model is still loading, please wait...",
                    "original": input_text,
                    "source_language": "unknown",
                    "target_language": "unknown",
                    "confidence_score": 0.0,
                    "processing_time": time.time() - start_time,
                    "model_used": self.model_name
                }
        
        try:
            # Detect languages
            source_lang = self.detect_language(input_text)
            target_lang = 'en' if source_lang == 'zh' else 'zh'
            
            # Create prompt
            prompt = self.create_prompt(input_text, source_lang, target_lang)
            
            # Generate translation
            outputs = self.llm.generate([prompt], self.sampling_params)
            translation = outputs[0].outputs[0].text.strip()
            
            # Clean up translation
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            
            # Remove common artifacts
            translation = translation.replace('Translation:', '').strip()
            translation = translation.replace('翻译：', '').strip()
            
            processing_time = time.time() - start_time
            
            return {
                "translation": translation,
                "original": input_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "confidence_score": 0.9,  # Simplified confidence
                "processing_time": processing_time,
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "translation": f"Translation error: {str(e)}",
                "original": input_text,
                "source_language": "unknown",
                "target_language": "unknown",
                "confidence_score": 0.0,
                "processing_time": time.time() - start_time,
                "model_used": self.model_name,
                "error": str(e)
            }
    
    async def register(self, websocket):
        """Register WebSocket client"""
        self.clients.add(websocket)
        self.stats['clients_connected'] = len(self.clients)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister(self, websocket):
        """Unregister WebSocket client"""
        self.clients.discard(websocket)
        self.stats['clients_connected'] = len(self.clients)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket):
        """Handle WebSocket client"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "text" in data:
                        input_text = data["text"]
                        logger.info(f"Translating: {input_text[:50]}...")
                        
                        # Process translation in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, 
                            self.translate_text,
                            input_text
                        )
                        
                        # Add WebSocket-specific metadata
                        result['session_id'] = data.get('session_id')
                        result['timestamp'] = datetime.utcnow().isoformat()
                        
                        # Send response
                        await websocket.send(json.dumps(result))
                        
                        # Update stats
                        self.stats['translations_processed'] += 1
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def serve(self):
        """Start WebSocket server"""
        # Note: HTTP server runs in separate thread
        async with websockets.serve(self.handle_client, self.host, self.port + 1):  # Use port+1 for WebSocket
            logger.info(f"🚀 WebSocket server running at ws://{self.host}:{self.port + 1}")
            logger.info("Waiting for model to load...")
            await asyncio.Future()  # Run forever

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple vLLM Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to (WebSocket will use port+1)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct-AWQ", help="Model name")
    
    args = parser.parse_args()
    
    server = SimpleVLLMTranslationServer(
        host=args.host,
        port=args.port,
        model_name=args.model
    )
    
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped")

if __name__ == "__main__":
    main() 