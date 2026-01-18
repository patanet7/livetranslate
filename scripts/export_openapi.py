#!/usr/bin/env python3
"""
Export OpenAPI specifications from FastAPI services.

Usage:
    python scripts/export_openapi.py [--output-dir docs/api]
"""

import argparse
import json
import sys
from pathlib import Path


def export_orchestration_openapi(output_dir: Path) -> bool:
    """Export OpenAPI spec from orchestration service."""
    try:
        # Add service to path
        service_path = Path(__file__).parent.parent / "modules" / "orchestration-service"
        sys.path.insert(0, str(service_path))

        from src.main_fastapi import app

        openapi_spec = app.openapi()
        output_file = output_dir / "orchestration-openapi.json"
        output_file.write_text(json.dumps(openapi_spec, indent=2))
        print(f"Exported: {output_file}")
        return True
    except ImportError as e:
        print(f"Warning: Could not import orchestration service: {e}")
        return False


def export_translation_openapi(output_dir: Path) -> bool:
    """Export OpenAPI spec from translation service."""
    try:
        service_path = Path(__file__).parent.parent / "modules" / "translation-service"
        sys.path.insert(0, str(service_path))

        from src.api_server import app

        openapi_spec = app.openapi()
        output_file = output_dir / "translation-openapi.json"
        output_file.write_text(json.dumps(openapi_spec, indent=2))
        print(f"Exported: {output_file}")
        return True
    except ImportError as e:
        print(f"Warning: Could not import translation service: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export OpenAPI specifications")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/api"),
        help="Output directory for OpenAPI specs",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(("orchestration", export_orchestration_openapi(args.output_dir)))
    results.append(("translation", export_translation_openapi(args.output_dir)))

    # Summary
    print("\nExport Summary:")
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()
