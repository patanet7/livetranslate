"""Transcription service entry point."""
import argparse
from pathlib import Path

import uvicorn
from livetranslate_common.logging import setup_logging

from api import create_app


def main():
    parser = argparse.ArgumentParser(description="Transcription Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "model_registry.yaml",
    )
    parser.add_argument("--log-format", default="dev", choices=["dev", "json"])
    args = parser.parse_args()

    setup_logging(service_name="transcription", log_format=args.log_format)

    app = create_app(registry_path=args.registry)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
