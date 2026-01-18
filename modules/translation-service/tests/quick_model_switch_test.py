#!/usr/bin/env python3
"""
Quick Model Switching Test - Run after restarting translation service

Usage:
    # Restart translation service first:
    cd modules/translation-service
    python src/api_server_fastapi.py

    # Then run this test:
    python tests/quick_model_switch_test.py

This tests:
1. Translation with initial model
2. Get model status
3. Switch to different model
4. Translation with new model
5. Compare results
"""

import asyncio
import time

import aiohttp

BASE_URL = "http://localhost:5003"


async def main():
    print("=" * 70)
    print("QUICK MODEL SWITCHING TEST")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        # Step 1: Health check
        print("\n1️⃣  Checking service health...")
        async with session.get(f"{BASE_URL}/health") as resp:
            health = await resp.json()
            print(f"   Status: {health.get('status')}")
            if health.get("status") != "healthy":
                print("   ❌ Service not healthy!")
                return

        # Step 2: Get model status
        print("\n2️⃣  Getting model status...")
        async with session.get(f"{BASE_URL}/api/models/status") as resp:
            if resp.status == 404:
                print("   ❌ Model status endpoint not found!")
                print("   👉 Did you restart the translation service after updating the code?")
                print(
                    "   👉 Run: cd modules/translation-service && python src/api_server_fastapi.py"
                )
                return

            status = await resp.json()
            current_model = status.get("current_model")
            print(f"   Current model: {current_model}")
            print(f"   Backend: {status.get('current_backend')}")
            print(f"   Ready: {status.get('is_ready')}")
            print(f"   Cached models: {status.get('cache_size')}")

        # Step 3: Translate with initial model
        print("\n3️⃣  Translating with current model...")
        test_text = "The server processes requests asynchronously using event loops."
        payload = {"text": test_text, "source_language": "en", "target_language": "es"}

        async with session.post(f"{BASE_URL}/api/translate", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                translation1 = result.get("translated_text", "")
                print(f"   EN: {test_text}")
                print(f"   ES: {translation1}")
                print(f"   Model: {result.get('model_used', current_model)}")
            else:
                print(f"   ❌ Translation failed: {await resp.text()}")
                return

        # Step 4: List available models
        print("\n4️⃣  Getting available models...")
        async with session.get(f"{BASE_URL}/api/models/list/ollama") as resp:
            if resp.status == 200:
                models_data = await resp.json()
                models = models_data.get("models", [])
                print(f"   Found {len(models)} models")
                if models:
                    print(f"   Available: {models[:5]}...")

                    # Find a different model
                    other_model = None
                    for m in models:
                        if m != current_model:
                            other_model = m
                            break

                    if other_model:
                        # Step 5: Switch model
                        print(f"\n5️⃣  Switching model: {current_model} → {other_model}")
                        switch_start = time.time()

                        async with session.post(
                            f"{BASE_URL}/api/models/switch",
                            json={"model": other_model, "backend": "ollama"},
                        ) as resp:
                            switch_result = await resp.json()
                            switch_time = time.time() - switch_start

                            if switch_result.get("success"):
                                print(f"   ✅ Switch successful in {switch_time:.2f}s")

                                # Step 6: Translate with new model
                                print(f"\n6️⃣  Translating with NEW model ({other_model})...")
                                async with session.post(
                                    f"{BASE_URL}/api/translate", json=payload
                                ) as resp:
                                    if resp.status == 200:
                                        result = await resp.json()
                                        translation2 = result.get("translated_text", "")
                                        print(f"   EN: {test_text}")
                                        print(f"   ES: {translation2}")
                                        print(f"   Model: {result.get('model_used', other_model)}")

                                        # Compare
                                        print("\n7️⃣  Comparing translations:")
                                        print(
                                            f"   Model 1 ({current_model}): {translation1[:60]}..."
                                        )
                                        print(f"   Model 2 ({other_model}): {translation2[:60]}...")

                                        if translation1 != translation2:
                                            print(
                                                "   ✅ Translations DIFFER - model switch confirmed!"
                                            )
                                        else:
                                            print(
                                                "   ⚠️ Translations identical (models may produce same output)"
                                            )

                                        # Switch back
                                        print(f"\n8️⃣  Switching back to {current_model}...")
                                        async with session.post(
                                            f"{BASE_URL}/api/models/switch",
                                            json={"model": current_model, "backend": "ollama"},
                                        ) as resp:
                                            back_result = await resp.json()
                                            if back_result.get("success"):
                                                print("   ✅ Switched back successfully")
                            else:
                                print(f"   ❌ Switch failed: {switch_result.get('message')}")
                    else:
                        print("   ⚠️ No other model available to switch to")
            else:
                print(f"   ❌ Could not list models: {await resp.text()}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
