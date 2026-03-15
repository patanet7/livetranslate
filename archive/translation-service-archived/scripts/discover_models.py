#!/usr/bin/env python3
"""
Model Discovery Utility

Queries Ollama, vLLM, and Open WebUI servers to discover available models.
Helps configure the translation service with the right model names.

Usage:
    python scripts/discover_models.py
    python scripts/discover_models.py --host 192.168.1.239
"""

import argparse
import asyncio
import json

import httpx


async def discover_ollama_models(host: str, port: int = 11434) -> dict | None:
    """
    Discover models from Ollama server.

    Args:
        host: Ollama server hostname/IP
        port: Ollama server port (default: 11434)

    Returns:
        Dict with status and models list
    """
    url = f"http://{host}:{port}/api/tags"
    print(f"\n🔍 Querying Ollama at {url}...")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON response: {e}")
                    print(f"   Response text: {response.text[:200]}")
                    return {"status": "error", "url": url, "models": []}

                models = data.get("models", [])

                print("✅ Ollama server is online")
                print(f"📋 Found {len(models)} model(s):\n")

                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0) / (1024**3)  # Convert to GB
                    modified = model.get("modified_at", "Unknown")

                    print(f"  • {name}")
                    print(f"    Size: {size:.2f} GB")
                    print(f"    Modified: {modified}")
                    print()

                return {
                    "status": "online",
                    "url": url,
                    "models": [m["name"] for m in models],
                    "count": len(models),
                }
            else:
                print(f"❌ Ollama server returned status {response.status_code}")
                print(f"   Response: {response.text}")
                return {"status": "error", "url": url, "models": []}

    except httpx.ConnectError:
        print(f"❌ Cannot connect to Ollama at {url}")
        print("   Make sure Ollama is running and accessible")
        return {"status": "offline", "url": url, "models": []}
    except Exception as e:
        print(f"❌ Error querying Ollama: {e}")
        return {"status": "error", "url": url, "models": []}


async def discover_vllm_models(host: str, port: int = 8092) -> dict | None:
    """
    Discover models from vLLM server (OpenAI-compatible).

    Args:
        host: vLLM server hostname/IP
        port: vLLM server port (default: 8092)

    Returns:
        Dict with status and models list
    """
    url = f"http://{host}:{port}/v1/models"
    print(f"\n🔍 Querying vLLM at {url}...")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                print("✅ vLLM server is online")
                print(f"📋 Found {len(models)} model(s):\n")

                for model in models:
                    model_id = model.get("id", "Unknown")
                    created = model.get("created", 0)
                    owned_by = model.get("owned_by", "Unknown")

                    print(f"  • {model_id}")
                    print(f"    Owner: {owned_by}")
                    print(f"    Created: {created}")
                    print()

                return {
                    "status": "online",
                    "url": url,
                    "models": [m["id"] for m in models],
                    "count": len(models),
                }
            else:
                print(f"❌ vLLM server returned status {response.status_code}")
                print(f"   Response: {response.text}")
                return {"status": "error", "url": url, "models": []}

    except httpx.ConnectError:
        print(f"❌ Cannot connect to vLLM at {url}")
        print("   Make sure vLLM server is running and accessible")
        return {"status": "offline", "url": url, "models": []}
    except Exception as e:
        print(f"❌ Error querying vLLM: {e}")
        return {"status": "error", "url": url, "models": []}


async def discover_openwebui(host: str, port: int = 3030) -> dict | None:
    """
    Check Open WebUI status.

    Args:
        host: Open WebUI hostname/IP
        port: Open WebUI port (default: 3030)

    Returns:
        Dict with status
    """
    url = f"http://{host}:{port}"
    print(f"\n🔍 Checking Open WebUI at {url}...")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code == 200:
                print(f"✅ Open WebUI is online at {url}")
                print("   Open WebUI provides a web interface for Ollama/OpenAI models")
                print("   It uses the same models as Ollama backend")

                return {
                    "status": "online",
                    "url": url,
                }
            else:
                print(f"❌ Open WebUI returned status {response.status_code}")
                return {"status": "error", "url": url}

    except httpx.ConnectError:
        print(f"❌ Cannot connect to Open WebUI at {url}")
        return {"status": "offline", "url": url}
    except Exception as e:
        print(f"❌ Error checking Open WebUI: {e}")
        return {"status": "error", "url": url}


def generate_env_config(ollama_result: dict, vllm_result: dict, host: str):
    """Generate .env configuration based on discovered models"""

    print("\n" + "=" * 80)
    print("📝 RECOMMENDED .ENV CONFIGURATION")
    print("=" * 80 + "\n")

    print("# Copy this to your .env file:\n")

    # Ollama configuration
    if ollama_result.get("status") == "online" and ollama_result.get("models"):
        print("# ============================================================================")
        print("# OLLAMA (Local/Remote)")
        print("# ============================================================================")
        print("OLLAMA_ENABLE=true")
        print(f"OLLAMA_BASE_URL=http://{host}:11434/v1")

        # Recommend first model or a good default
        models = ollama_result["models"]
        recommended_model = models[0]

        # Prefer llama models if available
        for model in models:
            if "llama" in model.lower():
                recommended_model = model
                break

        print(f"OLLAMA_MODEL={recommended_model}")
        print(f"\n# Available Ollama models: {', '.join(models)}")
        print()

    # vLLM configuration
    if vllm_result.get("status") == "online" and vllm_result.get("models"):
        print("# ============================================================================")
        print("# vLLM SERVER")
        print("# ============================================================================")
        print("VLLM_SERVER_ENABLE=true")
        print(f"VLLM_SERVER_BASE_URL=http://{host}:8092/v1")

        models = vllm_result["models"]
        recommended_model = models[0]

        print(f"VLLM_SERVER_MODEL={recommended_model}")
        print(f"\n# Available vLLM models: {', '.join(models)}")
        print()

    # Translation service configuration
    print("# ============================================================================")
    print("# TRANSLATION SERVICE")
    print("# ============================================================================")
    print("PORT=5003")
    print()

    print("# ============================================================================")
    print("# TRANSLATION CACHE (requires Redis)")
    print("# ============================================================================")
    print("REDIS_URL=redis://localhost:6379/1")
    print("TRANSLATION_CACHE_ENABLED=true")
    print("TRANSLATION_CACHE_TTL=3600")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Discover available models on Ollama/vLLM servers")
    parser.add_argument(
        "--host", default="192.168.1.239", help="Server hostname/IP (default: 192.168.1.239)"
    )
    parser.add_argument(
        "--ollama-port", type=int, default=11434, help="Ollama port (default: 11434)"
    )
    parser.add_argument("--vllm-port", type=int, default=8092, help="vLLM port (default: 8092)")
    parser.add_argument(
        "--openwebui-port", type=int, default=3030, help="Open WebUI port (default: 3030)"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    print("=" * 80)
    print("🔍 MODEL DISCOVERY UTILITY")
    print("=" * 80)
    print(f"\nScanning server: {args.host}")
    print(f"Ollama port: {args.ollama_port}")
    print(f"vLLM port: {args.vllm_port}")
    print(f"Open WebUI port: {args.openwebui_port}")

    # Discover models from all services
    ollama_result = await discover_ollama_models(args.host, args.ollama_port)
    vllm_result = await discover_vllm_models(args.host, args.vllm_port)
    openwebui_result = await discover_openwebui(args.host, args.openwebui_port)

    # Output results
    if args.json:
        results = {
            "host": args.host,
            "ollama": ollama_result,
            "vllm": vllm_result,
            "openwebui": openwebui_result,
        }
        print(json.dumps(results, indent=2))
    else:
        # Generate recommended configuration
        generate_env_config(ollama_result, vllm_result, args.host)

        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80 + "\n")

        total_models = 0

        if ollama_result.get("status") == "online":
            count = ollama_result.get("count", 0)
            total_models += count
            print(f"✅ Ollama: {count} model(s) available")

        if vllm_result.get("status") == "online":
            count = vllm_result.get("count", 0)
            total_models += count
            print(f"✅ vLLM Server: {count} model(s) available")

        if openwebui_result.get("status") == "online":
            print("✅ Open WebUI: Online")

        print(f"\n🎯 Total models discovered: {total_models}")

        if total_models == 0:
            print("\n⚠️  No models found. Please check:")
            print("   1. Are the servers running?")
            print("   2. Are the ports correct?")
            print("   3. Is the host accessible from this machine?")
            print("\n💡 Test connectivity:")
            print(f"   curl http://{args.host}:{args.ollama_port}/api/tags")
            print(f"   curl http://{args.host}:{args.vllm_port}/v1/models")

        print()


if __name__ == "__main__":
    asyncio.run(main())
