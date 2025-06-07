import asyncio
import websockets
import json
import logging
import argparse
import traceback
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationServer:
    def __init__(self, host="localhost", port=8010, model_name="Qwen/Qwen3-14B-AWQ"):
        self.host = host
        self.port = port
        self.clients = set()
        self.model_name = model_name
        self.lock = threading.Lock()
        self.is_model_ready = False
        
        # Initialize model in a separate thread to avoid blocking
        self.init_thread = threading.Thread(target=self._initialize_model)
        self.init_thread.daemon = True
        self.init_thread.start()
        
    def _initialize_model(self):
        """Initialize the tokenizer and model in a separate thread"""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Optimized sampling parameters for translation
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_tokens=300,
                presence_penalty=1.5,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )
            
            logger.info(f"Loading model: {self.model_name}")
            self.llm = LLM(
                model=self.model_name,
                quantization="awq_marlin",
                dtype="half",
                tensor_parallel_size=1,
                max_model_len=512,
            )
            
            # System prompt for translation
            self.system_prompt = (
                "You are an expert bilingual translator who fluently translates between Chinese and English.\n"
                "If the input is in Chinese characters (Hanzi), translate to English.\n"
                "If the input is in English, translate to Chinese.\n"
                "Avoid explanations. Only return the translated sentence with no formatting.\n"
                "Translate the following text:\n"
            )
            
            with self.lock:
                self.is_model_ready = True
            
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            traceback.print_exc()
    
    def get_prompt(self, input_text):
        """Create a prompt for the model"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    
    def translate_text(self, input_text):
        """Translate text using the LLM"""
        with self.lock:
            if not self.is_model_ready:
                return "Translation model is still loading, please wait..."
            
            try:
                # Detect if input is Chinese or English based on character set
                # This is a simplified approach - a more robust method would be better for production
                is_chinese = any('\u4e00' <= char <= '\u9fff' for char in input_text)
                
                # Generate the prompt
                prompt = self.get_prompt(input_text)
                
                # Generate translation
                outputs = self.llm.generate(prompt, sampling_params=self.sampling_params)
                translation = outputs[0].outputs[0].text.strip()
                
                # Simple post-processing to remove any artifacts
                # This could be enhanced for better quality
                if translation.startswith('"') and translation.endswith('"'):
                    translation = translation[1:-1]
                
                return translation
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return f"Error translating text: {str(e)}"
    
    async def register(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket):
        """Handle a client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse incoming message
                    data = json.loads(message)
                    
                    # Handle text to translate
                    if "text" in data:
                        input_text = data["text"]
                        logger.info(f"Received text to translate: {input_text[:30]}...")
                        
                        # Process in a thread pool to avoid blocking the event loop
                        loop = asyncio.get_event_loop()
                        translation = await loop.run_in_executor(
                            None, 
                            self.translate_text,
                            input_text
                        )
                        
                        # Send back translation
                        response = {
                            "translation": translation,
                            "original": input_text
                        }
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    traceback.print_exc()
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def serve(self):
        """Start the WebSocket server"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Translation server started at ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveTranslate Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind the server to")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-AWQ", help="Model name to use for translation")
    
    args = parser.parse_args()
    
    server = TranslationServer(host=args.host, port=args.port, model_name=args.model)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped") 