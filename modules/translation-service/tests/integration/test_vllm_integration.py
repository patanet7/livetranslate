#!/usr/bin/env python3
"""
Simple vLLM Translation Server Test

Test script to check basic functionality without heavy dependencies.
This will help us identify what's available and what needs to be installed.
"""

import sys
from datetime import datetime


def check_dependencies():
    """Check which dependencies are available"""
    print("=== Checking Dependencies ===")

    dependencies = [
        "flask",
        "flask_cors",
        "flask_socketio",
        "redis",
        "requests",
        "numpy",
        "transformers",
        "torch",
        "vllm",
        "langdetect",
        "websockets",
        "asyncio",
    ]

    available = []
    missing = []

    for dep in dependencies:
        try:
            __import__(dep)
            available.append(dep)
            print(f"✓ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep}")

    print(f"\nAvailable: {len(available)}/{len(dependencies)}")
    print(f"Missing: {missing}")

    return available, missing


def test_basic_server():
    """Test basic Flask server without vLLM"""
    print("\n=== Testing Basic Flask Server ===")

    try:
        from flask import Flask, jsonify, request
        from flask_cors import CORS

        app = Flask(__name__)
        CORS(app)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "model_ready": False,
                    "message": "Basic server running (no vLLM)",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        @app.route("/translate", methods=["POST"])
        def translate():
            data = request.get_json()
            text = data.get("text", "")

            # Mock translation for testing
            if any("\u4e00" <= c <= "\u9fff" for c in text):
                # Chinese to English (mock)
                translation = f"[EN] {text}"
                source_lang = "zh"
                target_lang = "en"
            else:
                # English to Chinese (mock)
                translation = f"[中文] {text}"
                source_lang = "en"
                target_lang = "zh"

            return jsonify(
                {
                    "original_text": text,
                    "translated_text": translation,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "confidence_score": 0.95,
                    "processing_time": 0.001,
                    "model_used": "mock_translator",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        print("✓ Basic Flask server created successfully")
        print("✓ Health endpoint: /health")
        print("✓ Translation endpoint: /translate")
        print("\nTo test, run:")
        print("python test_vllm_simple.py --run-server")

        return app

    except ImportError as e:
        print(f"✗ Failed to create Flask server: {e}")
        return None


def test_language_detection():
    """Test language detection without heavy dependencies"""
    print("\n=== Testing Language Detection ===")

    def simple_detect(text):
        """Simple Chinese character detection"""
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total_chars = len([c for c in text if c.isalnum()])

        if total_chars == 0:
            return "en"

        if chinese_chars / total_chars > 0.3:
            return "zh"
        else:
            return "en"

    test_cases = [
        "Hello, how are you?",
        "\u4f60\u597d\uff0c\u4f60\u597d\u5417\uff1f",  # Chinese: Hello, how are you?
        "Hello \u4f60\u597d",  # Mixed: Hello + Chinese hello
        "123 test",
        "\u6d4b\u8bd5 test \u6df7\u5408",  # Chinese-English mixed
    ]

    for text in test_cases:
        detected = simple_detect(text)
        print(f"'{text}' -> {detected}")

    print("✓ Basic language detection working")


def test_websocket_mock():
    """Test WebSocket functionality if available"""
    print("\n=== Testing WebSocket Support ===")

    try:
        import importlib.util
        import json

        if importlib.util.find_spec("websockets") is None:
            raise ImportError("websockets not installed")

        async def mock_websocket_handler(websocket, path):
            print(f"WebSocket client connected: {path}")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "translate":
                        response = {
                            "type": "translation",
                            "original_text": data.get("text", ""),
                            "translated_text": f"[MOCK] {data.get('text', '')}",
                            "source_language": "auto",
                            "target_language": "auto",
                            "confidence_score": 0.95,
                            "processing_time": 0.001,
                            "timestamp": datetime.now().isoformat(),
                        }
                        await websocket.send(json.dumps(response))
                except Exception as e:
                    print(f"WebSocket error: {e}")

        print("✓ WebSocket support available")
        print("✓ Mock WebSocket handler created")

        return mock_websocket_handler

    except ImportError as e:
        print(f"✗ WebSocket not available: {e}")
        return None


def run_basic_server():
    """Run the basic server for testing"""
    app = test_basic_server()
    if app:
        print("\n=== Starting Basic Server ===")
        print("Server will run on http://localhost:5003")
        print("Test with:")
        print(
            "curl -X POST http://localhost:5003/translate -H 'Content-Type: application/json' -d '{\"text\":\"Hello world\"}'"
        )
        print("\nPress Ctrl+C to stop")

        try:
            app.run(host="0.0.0.0", port=5003, debug=True)
        except KeyboardInterrupt:
            print("\nServer stopped")


def main():
    """Main test function"""
    print("vLLM Translation Server - Dependency Check & Basic Test")
    print("=" * 60)

    # Check what's available
    available, missing = check_dependencies()

    # Test basic functionality
    test_language_detection()

    # Test Flask server creation
    test_basic_server()

    # Test WebSocket if available
    test_websocket_mock()

    print("\n=== Summary ===")
    print(f"Dependencies available: {len(available)}")
    print(f"Dependencies missing: {len(missing)}")

    if "flask" in available:
        print("✓ Can run basic REST API server")
    else:
        print("✗ Cannot run REST API server (Flask missing)")

    if "websockets" in available:
        print("✓ Can run WebSocket server")
    else:
        print("✗ Cannot run WebSocket server")

    if "vllm" in available and "torch" in available:
        print("✓ Can run full vLLM translation")
    else:
        print("✗ Cannot run vLLM translation (missing vLLM/torch)")
        print("  -> Can run with mock translation for testing")

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--run-server":
        run_basic_server()
    else:
        print(f"\nTo run basic server: python {sys.argv[0]} --run-server")


if __name__ == "__main__":
    main()
