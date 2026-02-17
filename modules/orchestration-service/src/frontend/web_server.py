#!/usr/bin/env python3
"""
Frontend Web Server Component

Provides the web interface, static file serving, and template rendering
for the orchestration service dashboard. Built upon the extracted frontend-service
functionality from the legacy whisper-npu-server.
"""

import os
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from livetranslate_common.logging import get_logger
from werkzeug.exceptions import NotFound

logger = get_logger()


class WebServer:
    """Web server component for frontend interface"""

    def __init__(self, config: dict[str, Any]):
        """Initialize web server with configuration"""
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 3000)
        self.workers = config.get("workers", 4)

        # Setup paths
        self.template_folder = Path(__file__).parent.parent.parent / "templates"
        self.static_folder = Path(__file__).parent.parent.parent / "static"

        # Ensure directories exist
        self.template_folder.mkdir(exist_ok=True)
        self.static_folder.mkdir(exist_ok=True)

        # Service status
        self.running = False
        self.start_time = 0

        # Application settings from frontend-service
        self.app_settings = {
            "app_name": "LiveTranslate Orchestration",
            "version": "2.0.0",
            "auto_refresh_interval": 30,
            "max_log_entries": 1000,
            "enable_analytics": True,
            "enable_pwa": True,
            "default_language": "en",
            "supported_languages": [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            "audio_settings": {
                "default_sample_rate": 16000,
                "supported_sample_rates": [16000, 22050, 44100, 48000],
                "default_chunk_duration": 1000,
                "max_recording_duration": 300,
            },
            "ui_settings": {
                "theme": "dark",
                "show_advanced_controls": False,
                "auto_scroll_transcripts": True,
                "show_confidence_scores": True,
                "show_timestamps": True,
            },
        }

        logger.info(f"Web server initialized on {self.host}:{self.port}")

    async def start(self):
        """Start the web server component"""
        self.running = True
        self.start_time = time.time()
        logger.info("Web server component started")

    async def stop(self):
        """Stop the web server component"""
        self.running = False
        logger.info("Web server component stopped")

    def render_dashboard(self) -> str:
        """Render the main dashboard"""
        try:
            # Dashboard configuration matching the extracted frontend
            dashboard_config = {
                "title": "LiveTranslate Orchestration Dashboard",
                "version": self.app_settings["version"],
                "app_settings": self.app_settings,
                "service_config": {
                    "whisper": {
                        "url": os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001"),
                        "health_endpoint": "/health",
                        "websocket_url": os.getenv("WHISPER_WS_URL", "ws://localhost:5001"),
                        "status": "unknown",
                    },
                    "translation": {
                        "url": os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003"),
                        "health_endpoint": "/api/health",
                        "websocket_url": os.getenv("TRANSLATION_WS_URL", "ws://localhost:5003"),
                        "status": "unknown",
                    },
                    "speaker": {
                        "url": os.getenv("SPEAKER_SERVICE_URL", "http://localhost:5002"),
                        "health_endpoint": "/health",
                        "websocket_url": os.getenv("SPEAKER_WS_URL", "ws://localhost:5002"),
                        "status": "unknown",
                    },
                },
                "features": {
                    "real_time_monitoring": True,
                    "service_health": True,
                    "websocket_management": True,
                    "performance_analytics": True,
                    "session_management": True,
                    "real_time_transcription": True,
                    "speaker_diarization": True,
                    "translation": True,
                    "audio_testing": True,
                    "export_capabilities": True,
                },
                "ui_settings": {
                    "theme": "dark",
                    "auto_refresh": True,
                    "refresh_interval": 5000,
                    "show_advanced_controls": False,
                },
            }

            return render_template(
                "dashboard.html",
                config=dashboard_config,
                app_settings=self.app_settings,
            )

        except Exception as e:
            logger.error(f"Dashboard rendering failed: {e}")
            return self._render_error_page("Dashboard Error", str(e))

    def render_websocket_test(self) -> str:
        """Render WebSocket test page"""
        try:
            return render_template("websocket-test.html", app_settings=self.app_settings)
        except Exception as e:
            logger.error(f"WebSocket test rendering failed: {e}")
            return self._render_error_page("WebSocket Test Error", str(e))

    def render_pipeline_test(self) -> str:
        """Render pipeline test page"""
        try:
            return render_template("pipeline-test.html", app_settings=self.app_settings)
        except Exception as e:
            logger.error(f"Pipeline test rendering failed: {e}")
            return self._render_error_page("Pipeline Test Error", str(e))

    def render_audio_test(self) -> str:
        """Render audio test page"""
        try:
            return render_template("audio-test-consolidated.html", app_settings=self.app_settings)
        except Exception as e:
            logger.error(f"Audio test rendering failed: {e}")
            return self._render_error_page("Audio Test Error", str(e))

    def render_settings(self) -> str:
        """Render settings page"""
        try:
            return render_template("settings.html", app_settings=self.app_settings)
        except Exception as e:
            logger.error(f"Settings rendering failed: {e}")
            return self._render_error_page("Settings Error", str(e))

    def render_404(self) -> str:
        """Render 404 page"""
        try:
            return render_template("404.html", app_settings=self.app_settings)
        except Exception as e:
            logger.error(f"404 rendering failed: {e}")
            return self._render_error_page(
                "404 Not Found", "The requested page could not be found."
            )

    def serve_static_file(self, filename: str):
        """Serve static files"""
        try:
            return send_from_directory(self.static_folder, filename)
        except NotFound:
            logger.warning(f"Static file not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"Static file serving error: {e}")
            raise

    def get_frontend_config(self) -> dict[str, Any]:
        """Get frontend configuration for API"""
        return {
            "services": {
                "whisper": {
                    "url": os.getenv("WHISPER_SERVICE_URL", "http://localhost:5001"),
                    "features": ["transcription", "streaming", "models"],
                },
                "speaker": {
                    "url": os.getenv("SPEAKER_SERVICE_URL", "http://localhost:5002"),
                    "features": ["diarization", "clustering", "continuity"],
                },
                "translation": {
                    "url": os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003"),
                    "features": [
                        "translation",
                        "language_detection",
                        "local_inference",
                    ],
                },
                "orchestration": {
                    "url": f"http://localhost:{self.port}",
                    "features": [
                        "routing",
                        "load_balancing",
                        "monitoring",
                        "session_management",
                    ],
                },
            },
            "ui": {
                "theme": os.getenv("UI_THEME", self.app_settings["ui_settings"]["theme"]),
                "language": os.getenv("UI_LANGUAGE", self.app_settings["default_language"]),
                "audio_visualization": True,
                "speaker_colors": [
                    "#FF6B6B",
                    "#4ECDC4",
                    "#45B7D1",
                    "#96CEB4",
                    "#FFEAA7",
                    "#DDA0DD",
                    "#98D8C8",
                    "#F7DC6F",
                    "#BB8FCE",
                    "#85C1E9",
                ],
            },
            "features": self.app_settings,
            "audio_settings": self.app_settings["audio_settings"],
        }

    def update_app_settings(self, new_settings: dict[str, Any]) -> dict[str, Any]:
        """Update application settings"""
        updated = {}

        for key, value in new_settings.items():
            if key in self.app_settings:
                self.app_settings[key] = value
                updated[key] = value

        return updated

    def _render_error_page(self, title: str, error: str) -> str:
        """Render error page"""
        try:
            return render_template(
                "error.html", title=title, error=error, app_settings=self.app_settings
            )
        except Exception:
            # Fallback HTML if template rendering fails
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Inter', Arial, sans-serif;
                        margin: 40px;
                        background: #1a1a1a;
                        color: #e0e0e0;
                    }}
                    .error {{
                        color: #ff6b6b;
                        background: #2d1b1b;
                        padding: 20px;
                        border-radius: 8px;
                        border-left: 4px solid #ff6b6b;
                    }}
                    h1 {{ color: #4ECDC4; }}
                </style>
            </head>
            <body>
                <h1>LiveTranslate Orchestration Service</h1>
                <div class="error">
                    <h2>{title}</h2>
                    <p>{error}</p>
                </div>
            </body>
            </html>
            """

    def get_status(self) -> dict[str, Any]:
        """Get web server status"""
        return {
            "component": "web_server",
            "status": "running" if self.running else "stopped",
            "uptime": time.time() - self.start_time if self.running else 0,
            "configuration": {
                "host": self.host,
                "port": self.port,
                "workers": self.workers,
                "template_folder": str(self.template_folder),
                "static_folder": str(self.static_folder),
            },
            "app_settings": self.app_settings,
        }


def add_web_routes(app: Flask, web_server: WebServer):
    """Add web routes to Flask application"""

    # Main pages
    @app.route("/")
    def dashboard():
        """Main dashboard"""
        return web_server.render_dashboard()

    @app.route("/websocket-test")
    def websocket_test():
        """WebSocket test page"""
        return web_server.render_websocket_test()

    @app.route("/pipeline-test")
    def pipeline_test():
        """Pipeline test page"""
        return web_server.render_pipeline_test()

    @app.route("/audio-test")
    def audio_test():
        """Audio test page"""
        return web_server.render_audio_test()

    @app.route("/settings")
    def settings():
        """Settings page"""
        return web_server.render_settings()

    # Static file serving
    @app.route("/static/<path:filename>")
    def serve_static(filename):
        """Serve static files"""
        return web_server.serve_static_file(filename)

    # API endpoints for frontend configuration
    @app.route("/api/config", methods=["GET"])
    def get_frontend_config():
        """Get frontend configuration"""
        return jsonify(web_server.get_frontend_config())

    @app.route("/api/config", methods=["POST"])
    def update_frontend_config():
        """Update frontend configuration"""
        try:
            config_data = request.get_json()
            if not config_data:
                return jsonify({"error": "No configuration provided"}), 400

            updated = web_server.update_app_settings(config_data)

            return jsonify(
                {
                    "status": "success",
                    "message": "Configuration updated",
                    "updated": updated,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/debug/multipart", methods=["POST"])
    def debug_multipart():
        """Debug multipart form data handling"""
        try:
            logger.info("[DEBUG] Multipart debug endpoint called")
            logger.info(f"[DEBUG] Content-Type: {request.content_type}")
            logger.info(f"[DEBUG] Method: {request.method}")
            logger.info(f"[DEBUG] Files: {list(request.files.keys()) if request.files else 'None'}")
            logger.info(f"[DEBUG] Form: {list(request.form.keys()) if request.form else 'None'}")
            logger.info(f"[DEBUG] Data length: {len(request.data) if request.data else 0}")

            result = {
                "status": "success",
                "content_type": request.content_type,
                "files_received": [],
                "form_data": dict(request.form) if request.form else {},
                "data_length": len(request.data) if request.data else 0,
            }

            if request.files:
                for key, file in request.files.items():
                    file_size = 0
                    try:
                        file.seek(0)
                        file_content = file.read()
                        file_size = len(file_content)
                        file.seek(0)  # Reset for potential re-reading

                        result["files_received"].append(
                            {
                                "key": key,
                                "filename": file.filename,
                                "content_type": file.content_type,
                                "size": file_size,
                            }
                        )

                        logger.info(
                            f"[DEBUG] File {key}: {file.filename} ({file_size} bytes, {file.content_type})"
                        )

                    except Exception as file_error:
                        logger.error(f"[DEBUG] Error reading file {key}: {file_error}")
                        result["files_received"].append(
                            {
                                "key": key,
                                "filename": file.filename,
                                "content_type": file.content_type,
                                "size": file_size,
                                "error": str(file_error),
                            }
                        )

            logger.info(f"[DEBUG] Returning result: {result}")
            return jsonify(result)

        except Exception as e:
            logger.error(f"[DEBUG] Multipart debug failed: {e}")
            return jsonify({"error": str(e)}), 500

    # Settings management
    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        """Get current application settings"""
        return jsonify(
            {
                **web_server.app_settings,
                "runtime_info": {
                    "uptime": time.time() - web_server.start_time if web_server.running else 0,
                    "status": "running" if web_server.running else "stopped",
                },
            }
        )

    @app.route("/api/settings", methods=["POST"])
    def update_settings():
        """Update application settings"""
        try:
            new_settings = request.get_json()
            if not new_settings:
                return jsonify({"error": "No settings provided"}), 400

            updated = web_server.update_app_settings(new_settings)

            return jsonify({"message": "Settings updated successfully", "updated": updated})

        except Exception as e:
            logger.error(f"Settings update error: {e}")
            return jsonify({"error": str(e)}), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        if request.path.startswith("/api/"):
            return jsonify({"error": "API endpoint not found"}), 404
        else:
            return web_server.render_404(), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors"""
        logger.error(f"Internal server error: {error}")
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error"}), 500
        else:
            return (
                web_server._render_error_page(
                    "Internal Server Error", "An internal server error occurred."
                ),
                500,
            )
