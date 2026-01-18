#!/usr/bin/env python3
"""
Service Integration Module

Demonstrates integration between:
- Whisper Service → Translation Service
- Translation Service → Speaker Service (Post-processing)
- Frontend Service ← All Services

This shows the complete pipeline for live translation.
"""

import asyncio
import json
import logging
from datetime import datetime

import requests
import websockets

logger = logging.getLogger(__name__)


class ServiceIntegration:
    """Integration manager for translation service with other services"""

    def __init__(
        self,
        translation_host="localhost",
        translation_port=8010,
        whisper_host="localhost",
        whisper_port=5001,
        speaker_host="localhost",
        speaker_port=5002,
        frontend_host="localhost",
        frontend_port=3000,
    ):
        # Service endpoints
        self.services = {
            "translation": {
                "rest": f"http://{translation_host}:{translation_port}",
                "websocket": f"ws://{translation_host}:{translation_port + 1}",
            },
            "whisper": {
                "rest": f"http://{whisper_host}:{whisper_port}",
                "websocket": f"ws://{whisper_host}:{whisper_port}/ws",
            },
            "speaker": {
                "rest": f"http://{speaker_host}:{speaker_port}",
                "websocket": f"ws://{speaker_host}:{speaker_port}/ws",
            },
            "frontend": {
                "rest": f"http://{frontend_host}:{frontend_port}",
                "websocket": f"ws://{frontend_host}:{frontend_port}/ws",
            },
        }

        # WebSocket connections
        self.ws_connections = {}
        self.session_data = {}

        logger.info("🔗 Service Integration initialized")
        logger.info(f"Translation Service: {self.services['translation']['rest']}")
        logger.info(f"Whisper Service: {self.services['whisper']['rest']}")
        logger.info(f"Speaker Service: {self.services['speaker']['rest']}")
        logger.info(f"Frontend Service: {self.services['frontend']['rest']}")

    def check_service_health(self, service_name: str) -> dict:
        """Check if a service is healthy"""
        try:
            response = requests.get(f"{self.services[service_name]['rest']}/health", timeout=5)
            return {
                "service": service_name,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response": response.json() if response.status_code == 200 else None,
                "error": None,
            }
        except Exception as e:
            return {"service": service_name, "status": "error", "response": None, "error": str(e)}

    def check_all_services(self) -> dict:
        """Check health of all services"""
        results = {}
        for service_name in self.services:
            results[service_name] = self.check_service_health(service_name)

        all_healthy = all(r["status"] == "healthy" for r in results.values())

        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def translate_text(
        self, text: str, source_lang: str = "auto", target_lang: str = "auto"
    ) -> dict:
        """Translate text using translation service REST API"""
        try:
            data = {"text": text, "source_lang": source_lang, "target_lang": target_lang}

            response = requests.post(
                f"{self.services['translation']['rest']}/translate", json=data, timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            logger.error(f"Translation request failed: {e}")
            return {"error": str(e)}

    async def process_whisper_transcription(self, transcription_data: dict) -> dict:
        """Process transcription from whisper service and translate it"""
        try:
            # Extract transcription data
            text = transcription_data.get("text", "")
            language = transcription_data.get("language", "auto")
            session_id = transcription_data.get("session_id")
            timestamp = transcription_data.get("timestamp")
            speaker_id = transcription_data.get("speaker_id")

            logger.info(f"🎤 Processing transcription: {text[:50]}...")

            # Translate the text
            translation_result = await self.translate_text(
                text=text, source_lang=language, target_lang="en" if language == "zh" else "zh"
            )

            if "error" in translation_result:
                logger.error(f"Translation failed: {translation_result['error']}")
                return {
                    "status": "error",
                    "error": translation_result["error"],
                    "original_data": transcription_data,
                }

            # Combine transcription and translation data
            processed_data = {
                "session_id": session_id,
                "timestamp": timestamp,
                "speaker_id": speaker_id,
                "transcription": {"text": text, "language": language},
                "translation": {
                    "text": translation_result.get("translation", ""),
                    "source_language": translation_result.get("source_language"),
                    "target_language": translation_result.get("target_language"),
                    "confidence_score": translation_result.get("confidence_score"),
                    "processing_time": translation_result.get("processing_time"),
                },
                "processing_timestamp": datetime.utcnow().isoformat(),
                "status": "success",
            }

            logger.info(
                f"✅ Translation complete: {text[:30]} → {translation_result.get('translation', '')[:30]}"
            )

            # Send to speaker service for post-processing
            await self.send_to_speaker_service(processed_data)

            return processed_data

        except Exception as e:
            logger.error(f"Transcription processing error: {e}")
            return {"status": "error", "error": str(e), "original_data": transcription_data}

    async def send_to_speaker_service(self, processed_data: dict) -> bool:
        """Send processed data to speaker service for post-processing (alignment, etc.)"""
        try:
            # Prepare data for speaker service post-processing
            speaker_data = {
                "session_id": processed_data["session_id"],
                "timestamp": processed_data["timestamp"],
                "speaker_id": processed_data["speaker_id"],
                "original_text": processed_data["transcription"]["text"],
                "translated_text": processed_data["translation"]["text"],
                "source_language": processed_data["translation"]["source_language"],
                "target_language": processed_data["translation"]["target_language"],
                "processing_type": "alignment_and_combination",
                "metadata": {
                    "confidence_score": processed_data["translation"]["confidence_score"],
                    "processing_time": processed_data["translation"]["processing_time"],
                },
            }

            # Send to speaker service for post-processing
            response = requests.post(
                f"{self.services['speaker']['rest']}/process", json=speaker_data, timeout=10
            )

            if response.status_code == 200:
                logger.info("📤 Sent to speaker service for post-processing")

                # Send final result to frontend
                await self.send_to_frontend(processed_data)
                return True
            else:
                logger.error(f"Speaker service error: HTTP {response.status_code}")

                # Still send to frontend even if speaker service fails
                await self.send_to_frontend(processed_data)
                return False

        except Exception as e:
            logger.error(f"Speaker service integration error: {e}")

            # Send to frontend as fallback
            await self.send_to_frontend(processed_data)
            return False

    async def send_to_frontend(self, final_data: dict) -> bool:
        """Send final processed data to frontend service"""
        try:
            # Prepare data for frontend display
            frontend_data = {
                "type": "translation_result",
                "session_id": final_data["session_id"],
                "timestamp": final_data["timestamp"],
                "speaker_id": final_data.get("speaker_id"),
                "content": {
                    "original": {
                        "text": final_data["transcription"]["text"],
                        "language": final_data["transcription"]["language"],
                    },
                    "translation": {
                        "text": final_data["translation"]["text"],
                        "language": final_data["translation"]["target_language"],
                    },
                },
                "metadata": {
                    "confidence": final_data["translation"]["confidence_score"],
                    "processing_time": final_data["translation"]["processing_time"],
                    "timestamp": final_data["processing_timestamp"],
                },
            }

            # Send via WebSocket if available, otherwise REST
            if "frontend" in self.ws_connections:
                await self.ws_connections["frontend"].send(json.dumps(frontend_data))
                logger.info("📡 Sent to frontend via WebSocket")
            else:
                # Fallback to REST API
                response = requests.post(
                    f"{self.services['frontend']['rest']}/api/translation-result",
                    json=frontend_data,
                    timeout=5,
                )

                if response.status_code == 200:
                    logger.info("📡 Sent to frontend via REST")
                else:
                    logger.warning(f"Frontend REST failed: HTTP {response.status_code}")

            return True

        except Exception as e:
            logger.error(f"Frontend integration error: {e}")
            return False

    async def start_whisper_integration(self):
        """Start integration with whisper service via WebSocket"""
        try:
            whisper_ws_url = self.services["whisper"]["websocket"]
            logger.info(f"🔌 Connecting to Whisper service: {whisper_ws_url}")

            async with websockets.connect(whisper_ws_url) as websocket:
                self.ws_connections["whisper"] = websocket
                logger.info("✅ Connected to Whisper service")

                async for message in websocket:
                    try:
                        data = json.loads(message)

                        # Check if this is a transcription result
                        if data.get("type") == "transcription_result":
                            await self.process_whisper_transcription(data)
                        elif data.get("type") == "error":
                            logger.error(f"Whisper service error: {data.get('message')}")
                        else:
                            logger.debug(f"Received from Whisper: {data}")

                    except json.JSONDecodeError:
                        logger.error("Invalid JSON from Whisper service")
                    except Exception as e:
                        logger.error(f"Error processing Whisper message: {e}")

        except Exception as e:
            logger.error(f"Whisper integration failed: {e}")
        finally:
            if "whisper" in self.ws_connections:
                del self.ws_connections["whisper"]

    async def start_frontend_integration(self):
        """Start integration with frontend service via WebSocket"""
        try:
            frontend_ws_url = self.services["frontend"]["websocket"]
            logger.info(f"🔌 Connecting to Frontend service: {frontend_ws_url}")

            async with websockets.connect(frontend_ws_url) as websocket:
                self.ws_connections["frontend"] = websocket
                logger.info("✅ Connected to Frontend service")

                async for message in websocket:
                    try:
                        data = json.loads(message)

                        # Handle frontend requests
                        if data.get("type") == "translation_request":
                            # Direct translation request from frontend
                            text = data.get("text", "")
                            result = await self.translate_text(text)

                            response = {
                                "type": "translation_response",
                                "request_id": data.get("request_id"),
                                "result": result,
                            }

                            await websocket.send(json.dumps(response))

                    except json.JSONDecodeError:
                        logger.error("Invalid JSON from Frontend service")
                    except Exception as e:
                        logger.error(f"Error processing Frontend message: {e}")

        except Exception as e:
            logger.error(f"Frontend integration failed: {e}")
        finally:
            if "frontend" in self.ws_connections:
                del self.ws_connections["frontend"]

    async def run_integration_loop(self):
        """Run the main integration loop"""
        logger.info("🚀 Starting service integration loop")

        # Check service health
        health_status = self.check_all_services()
        logger.info(f"🔍 Service health check: {health_status['overall_status']}")

        for service, status in health_status["services"].items():
            if status["status"] == "healthy":
                logger.info(f"  ✅ {service}: healthy")
            else:
                logger.warning(
                    f"  ❌ {service}: {status['status']} - {status.get('error', 'Unknown error')}"
                )

        # Start integrations concurrently
        tasks = [
            asyncio.create_task(self.start_whisper_integration()),
            asyncio.create_task(self.start_frontend_integration()),
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Integration stopped by user")
        except Exception as e:
            logger.error(f"Integration error: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Translation Service Integration")
    parser.add_argument("--translation-host", default="localhost", help="Translation service host")
    parser.add_argument(
        "--translation-port", type=int, default=8010, help="Translation service port"
    )
    parser.add_argument("--whisper-host", default="localhost", help="Whisper service host")
    parser.add_argument("--whisper-port", type=int, default=5001, help="Whisper service port")
    parser.add_argument("--speaker-host", default="localhost", help="Speaker service host")
    parser.add_argument("--speaker-port", type=int, default=5002, help="Speaker service port")
    parser.add_argument("--frontend-host", default="localhost", help="Frontend service host")
    parser.add_argument("--frontend-port", type=int, default=3000, help="Frontend service port")
    parser.add_argument("--test-only", action="store_true", help="Only test service connectivity")

    args = parser.parse_args()

    # Initialize integration
    integration = ServiceIntegration(
        translation_host=args.translation_host,
        translation_port=args.translation_port,
        whisper_host=args.whisper_host,
        whisper_port=args.whisper_port,
        speaker_host=args.speaker_host,
        speaker_port=args.speaker_port,
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
    )

    if args.test_only:
        # Test mode - just check connectivity
        health_status = integration.check_all_services()
        print(json.dumps(health_status, indent=2))
        return

    # Run integration
    try:
        asyncio.run(integration.run_integration_loop())
    except KeyboardInterrupt:
        logger.info("👋 Integration stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
